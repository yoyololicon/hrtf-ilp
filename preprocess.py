import argparse
import numpy as np
import sound_field_analysis as sfa
from pathlib import Path
import yaml
from functools import partial, reduce
from itertools import starmap, product

from toa import smooth_toa


def main():
    parser = argparse.ArgumentParser("Calculate time of arrival from HRTF")
    parser.add_argument("input", help="Input sofa file")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--toa-weight", type=float, default=0.1)
    parser.add_argument("--oversampling", type=int, default=10)

    args = parser.parse_args()

    hrir = sfa.io.read_SOFA_file(args.input)
    sr = hrir.l.fs
    az = hrir.grid.azimuth
    col = hrir.grid.colatitude
    radius = hrir.grid.radius
    hrir_signal = np.stack((hrir.l.signal, hrir.r.signal), axis=1)
    hrir_xyz = sfa.utils.sph2cart((az, col, radius)).T

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def worker(
        method: str, ignore_toa: bool, ignore_cross: bool, weighting_method: str
    ):
        npz_filename = f"toa_{ignore_toa}_cross_{ignore_cross}_{weighting_method}.npz"

        toa, m_shape, elapsed_time = smooth_toa(
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
        )

    yaml.safe_dump(vars(args), open(out_dir / "args.yaml", "w"))

    list(
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


if __name__ == "__main__":
    main()
