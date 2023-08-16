import numpy as np
from tqdm import tqdm
import maxflow
import sound_field_analysis as sfa
import argparse

from graph import *
from rigid import *


def phase2delay(phase, freqs, sr):
    return -phase / (freqs / sr * 2 * np.pi)


def delay2phase(delay, freqs, sr):
    return -delay * (freqs / sr * 2 * np.pi)


def puma(psi, edges, max_jump=1, p=1, verbose=False):
    if max_jump > 1:
        jump_steps = list(range(1, max_jump + 1)) * 2
    else:
        jump_steps = [max_jump]

    total_nodes = psi.size
    t = total_nodes
    s = t + 1

    def V(x):
        return np.abs(x) ** p

    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        return np.sum(V(2 * np.pi * (K[j] - K[i]) - psi[i] + psi[j]))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    energy_list = []
    with tqdm(disable=not verbose) as pbar:
        for step in jump_steps:
            while 1:
                energy_list.append(prev_Ek)
                G = maxflow.Graph[float]()
                G.add_nodes(total_nodes)

                i, j = edges[:, 0], edges[:, 1]
                psi_diff = psi[i] - psi[j]
                a = 2 * np.pi * (K[j] - K[i]) - psi_diff
                e00 = e11 = V(a)
                e01 = V(a - 2 * np.pi * step)
                e10 = V(a + 2 * np.pi * step)
                weight = np.maximum(0, e10 + e01 - e00 - e11)

                G.add_edges(edges[:, 0], edges[:, 1], weight, np.zeros_like(weight))

                tmp_st_weight = np.zeros((2, total_nodes))

                for i in range(edges.shape[0]):
                    u, v = edges[i]
                    tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
                    tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
                    tmp_st_weight[1, u] -= min(0, e10[i] - e00[i])
                    tmp_st_weight[1, v] -= min(0, e11[i] - e10[i])

                for i in range(total_nodes):
                    G.add_tedge(i, tmp_st_weight[0, i], tmp_st_weight[1, i])

                G.maxflow()

                partition = G.get_grid_segments(np.arange(total_nodes))
                K[~partition] += step

                energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

                pbar.set_description(f"Energy: {energy:.2f}")
                pbar.update(1)

                if energy < prev_Ek:
                    prev_Ek = energy
                else:
                    K[~partition] -= step
                    break

    return psi + 2 * np.pi * K


def puma_hrtf_phase(wrapped_phase, edges, max_jump=1, p=1, **kwargs):
    assert wrapped_phase.ndim == 2

    unwrapped_phase = puma(
        wrapped_phase.flatten(), edges, max_jump, p=p, **kwargs
    ).reshape(wrapped_phase.shape)
    return unwrapped_phase


def unwrap(
    hrir: np.ndarray,
    xyz: np.ndarray,
    sr: int,
    method: str,
    equalize: bool,
    stereo_proj: bool = False,
    p: float = 1.0,
    verbose: bool = True,
) -> dict:
    N, _, n_fft = hrir.shape

    hrtf = np.fft.rfft(hrir, axis=2)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    phase = np.angle(hrtf[..., 1:])

    save_dict = {}

    if equalize:
        toa = hrtf_toa(hrir)
        params = get_rigid_params(toa, xyz, sr, verbose=verbose)
        save_dict.update(params)
        smoothed_toa = toa_model(xyz, sr=sr, **params)

        linear_phase = delay2phase(smoothed_toa[..., None], freqs[1:], sr)
        phase -= linear_phase
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

    if method == "naive":
        unwrapped_phase = np.unwrap(phase, axis=2)
    elif method == "maxflow":
        G = plus_freq_dim(points2graph(xyz, stereo_proj), freqs.size - 1)
        edges = np.array(G.edges)
        if verbose:
            print(f"Number of edges: {edges.shape[0]}")
        unwrapped_phase = np.stack(
            (
                puma_hrtf_phase(phase[:, 0, :].T, edges, p=p, verbose=verbose).T,
                puma_hrtf_phase(phase[:, 1, :].T, edges, p=p, verbose=verbose).T,
            ),
            axis=1,
        )
    else:
        raise ValueError(f"Unknown method {method}")

    if equalize:
        unwrapped_phase += linear_phase

    # callibrate phase
    left_offset = np.round(
        np.mean(unwrapped_phase[:, 0, 0] - phase[:, 0, 0]) / 2 / np.pi
    )
    unwrapped_phase[:, 0] -= left_offset * 2 * np.pi
    right_offset = np.round(
        np.mean(unwrapped_phase[:, 1, 0] - phase[:, 1, 0]) / 2 / np.pi
    )
    unwrapped_phase[:, 1] -= right_offset * 2 * np.pi

    phase_delay = phase2delay(unwrapped_phase, freqs[1:], sr)
    save_dict["magnitude"] = np.abs(hrtf)
    save_dict["phase_delay"] = phase_delay

    return save_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sofa_file", type=str, help="Path to SOFA file containing HRTF data."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output file containing the unwrapped phase and meta data.",
    )
    parser.add_argument(
        "--method", type=str, choices=["naive", "maxflow"], default="maxflow"
    )
    parser.add_argument(
        "--equalize", action="store_true", help="Equalize the phase of the HRTF."
    )
    parser.add_argument(
        "--stereo-proj",
        action="store_true",
        help="Use stereographic projection to build graph.",
    )
    parser.add_argument(
        "-p",
        type=float,
        default=1,
        help="Power of the cost function. Only used if method is maxflow.",
    )

    args = parser.parse_args()

    hrir = sfa.io.read_SOFA_file(args.sofa_file)
    sr = hrir.l.fs
    az = hrir.grid.azimuth
    col = hrir.grid.colatitude
    radius = hrir.grid.radius
    hrir_signal = np.stack((hrir.l.signal, hrir.r.signal), axis=1)
    hrir_xyz = sfa.utils.sph2cart((az, col, radius)).T

    save_dict = unwrap(
        hrir_signal,
        hrir_xyz,
        sr,
        method=args.method,
        equalize=args.equalize,
        stereo_proj=args.stereo_proj,
        p=args.p,
    )
    save_dict["coordinates"] = hrir_xyz
    save_dict["sr"] = sr

    np.savez(args.output_file, **save_dict)


if __name__ == "__main__":
    main()
