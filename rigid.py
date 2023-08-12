import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import sound_field_analysis as sfa
from functools import partial


__all__ = ["toa_model", "get_rigid_params", "hrtf_toa"]


def hrtf_toa(hrir: np.ndarray):
    hrtf = np.fft.fft(hrir, axis=-1)
    log_mag = np.log(hrtf).real
    min_phase = -np.imag(hilbert(log_mag))
    min_hrtf = np.exp(log_mag + 1j * min_phase)
    toa = np.argmax(np.fft.ifft(hrtf * min_hrtf.conj()).real, axis=-1)
    return toa


def toa_model(P, r, az0, az1, incli0, incli1, delta, sr):
    E = sfa.utils.sph2cart([[az0, az1], [incli0, incli1], [r, r]]).T
    R = np.sqrt(np.sum(P**2, 1))

    PE_cos = P @ E.T / R[:, None] / np.sqrt(np.sum(E**2, 1))
    path_length = np.where(
        PE_cos > 0, R[:, None] - r * PE_cos, R[:, None] + r * np.arcsin(-PE_cos)
    )
    y = path_length / 343 * sr + delta
    return y


def get_rigid_params(toa, xyz, sr):
    assert toa.shape[0] == xyz.shape[0]

    def toa_func(P, r, az0, az1, incli0, incli1, delta):
        return toa_model(P, r, az0, az1, incli0, incli1, delta, sr).ravel()

    popt, pcov = curve_fit(
        toa_func,
        xyz,
        toa.ravel(),
        p0=(0.0825, 0.5 * np.pi, 1.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0),
        bounds=(
            [0, 0, np.pi, 0, 0, -np.inf],
            [np.inf, np.pi, 2 * np.pi, np.pi, np.pi, np.inf],
        ),
    )
    rigid_toa = toa_func(xyz, *popt).reshape(toa.shape)
    delay = rigid_toa.mean() / sr - popt[-1] / sr
    perr = np.sqrt(np.diag(pcov))

    print(f"Rigid sphere radius: {popt[0] * 100} cm (error: {perr[0] * 100})")
    print(f"IR offset: {popt[-2] / sr * 1000} ms (error: {perr[-2] / sr * 1000})")
    print(
        f"Left ear position: {popt[1] / np.pi * 180}, {popt[3] / np.pi * 180} (az/co) (error: {perr[1] / np.pi * 180}, {perr[3] / np.pi * 180})"
    )
    print(
        f"Right ear position: {popt[2] / np.pi * 180 - 360}, {popt[4] / np.pi * 180} (az/co) (error: {perr[2] / np.pi * 180}, {perr[4] / np.pi * 180})",
    )
    print(f"Average delay: {delay * 1000} ms (error: {perr[-1] / sr * 1000})")

    return {
        k: v for k, v in zip(["r", "az0", "az1", "incli0", "incli1", "delta"], popt)
    }
