import argparse
import numpy as np

import sound_field_analysis as sfa
from pathlib import Path
import yaml
from functools import partial, reduce
from itertools import starmap, product
from tqdm import tqdm

from toa import smooth_toa


def calculate_noise_scaler(signal_power, noise_power, target_snr):
    target_noise_power = signal_power / (10 ** (target_snr / 10))
    return np.sqrt(np.maximum(0, target_noise_power - noise_power))


def main():
    parser = argparse.ArgumentParser("Calculate time of arrival from HRTF")
    parser.add_argument("input", help="Input sofa file")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--toa-weight", type=float, default=0.1)
    parser.add_argument("--oversampling", type=int, default=10)
    parser.add_argument("--snr", type=float, default=None)

    args = parser.parse_args()

    hrir = sfa.io.read_SOFA_file(args.input)
    sr = hrir.l.fs
    az = hrir.grid.azimuth
    col = hrir.grid.colatitude
    radius = hrir.grid.radius
    hrir_signal = np.stack((hrir.l.signal, hrir.r.signal), axis=1)
    hrir_xyz = sfa.utils.sph2cart((az, col, radius)).T

    if args.snr is not None:
        # find closest point to the frontal direction
        idx = np.argmin(
            np.abs(hrir_xyz / radius[:, None] - np.array([1, 0, 0])).sum(axis=-1)
        )
        ref_hrirs = hrir_signal[idx]

        power = ref_hrirs**2
        filtered_power = power[power > 0]
        # use the top 10% of the power as signal, and the last 10% as noise
        sorted_power = np.sort(filtered_power)
        n = sorted_power.shape[0]
        signal_power = np.mean(sorted_power[-int(n * 0.1) :])
        noise_power = np.mean(sorted_power[: int(n * 0.1)])
        inherent_snr = 10 * np.log10(signal_power / noise_power)
        print(f"Measured SNR: {inherent_snr} dB")

        noise_scaler = calculate_noise_scaler(signal_power, noise_power, args.snr)
        print(noise_scaler)
        hrir_signal = hrir_signal + np.random.randn(*hrir_signal.shape) * noise_scaler

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def worker(
        method: str, ignore_toa: bool, ignore_cross: bool, weighting_method: str
    ):
        npz_filename = f"{method}_toa_{not ignore_toa}_cross_{not ignore_cross}_{weighting_method}.npz"

        toa, m_shape, elapsed_time, num_edges, num_nodes = smooth_toa(
            hrir=hrir_signal,
            xyz=hrir_xyz,
            sr=sr,
            method=method,
            oversampling=args.oversampling,
            ignore_cross=ignore_cross,
            ignore_toa=ignore_toa,
            weighted=weighting_method != "none",
            weighting_method=weighting_method,
            toa_weight=args.toa_weight,
            verbose=False,
        )

        np.savez(
            out_dir / npz_filename,
            toa=toa,
            m_shape=m_shape,
            elapsed_time=elapsed_time,
            num_edges=num_edges,
            num_nodes=num_nodes,
        )

    yaml.safe_dump(vars(args), open(out_dir / "args.yaml", "w"))

    list(
        tqdm(
            starmap(
                worker,
                product(
                    ("ilp", "edgelist", "l2"),
                    (True, False),  # ignore_toa
                    (True, False),  # ignore_cross
                    ("none", "dot", "angle"),
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
