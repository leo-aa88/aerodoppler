"""Multiresolution STFT and noise-whitened time-frequency representation (SPEC §8).

Default settings for the 2 kHz analysis stream: Hann window 4.096 s (8192 samples), hop
0.25 s, one-sided PSD up to ``fmax`` (default 900 Hz). These are starting values, not
values confirmed by the source material; every parameter is exposed.

Two arrays are stored: ``power_db`` and ``whitened``. Whitening removes (1) the smooth
broadband noise floor along frequency (block-median background), (2) stationary spectral
lines along time (long-window per-bin median, which suppresses mains hum), and (3) scales
by a robust MAD estimate so values are comparable across recordings (0 = background,
1 = one robust standard deviation above it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, median_filter
from scipy.signal import ShortTimeFFT, get_window


@dataclass
class Spectrogram:
    times: NDArray[np.float64]  # (T,) seconds relative to audio start (frame centres)
    freqs: NDArray[np.float64]  # (F,) Hz
    power_db: NDArray[np.float32]  # (T, F)
    whitened: NDArray[np.float32]  # (T, F)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def hop_s(self) -> float:
        return float(self.times[1] - self.times[0]) if len(self.times) > 1 else 0.0

    @property
    def df_hz(self) -> float:
        return float(self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 else 0.0

    def sample_whitened(
        self, t_s: NDArray[np.float64], f_hz: NDArray[np.float64], fill: float = np.nan
    ) -> NDArray[np.float64]:
        """Bilinear interpolation of the whitened map at (t, f) pairs; outside -> fill."""
        ti = (np.asarray(t_s, dtype=np.float64) - self.times[0]) / self.hop_s
        fi = (np.asarray(f_hz, dtype=np.float64) - self.freqs[0]) / self.df_hz
        out = map_coordinates(self.whitened, [ti, fi], order=1, mode="constant", cval=fill)
        return np.asarray(out, dtype=np.float64)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            times=self.times,
            freqs=self.freqs,
            power_db=self.power_db,
            whitened=self.whitened,
            params=np.array([self.params], dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> Spectrogram:
        z = np.load(path, allow_pickle=True)
        params = z["params"][0] if "params" in z else {}
        return cls(z["times"], z["freqs"], z["power_db"], z["whitened"], dict(params))

    def crop(self, t0: float, t1: float, f0: float, f1: float) -> Spectrogram:
        ti = (self.times >= t0) & (self.times <= t1)
        fi = (self.freqs >= f0) & (self.freqs <= f1)
        return Spectrogram(
            self.times[ti],
            self.freqs[fi],
            self.power_db[np.ix_(ti, fi)],
            self.whitened[np.ix_(ti, fi)],
            dict(self.params),
        )


def compute_spectrogram(
    audio: NDArray[np.floating],
    fs: int,
    window_s: float = 4.096,
    hop_s: float = 0.25,
    fmax_hz: float = 900.0,
    whiten: bool = True,
    whiten_f_window_hz: float = 30.0,
    whiten_t_window_s: float = 600.0,
) -> Spectrogram:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("audio must be mono for spectrogram analysis")
    nwin = int(round(window_s * fs))
    hop = max(1, int(round(hop_s * fs)))
    if len(x) < nwin:
        raise ValueError(f"audio too short ({len(x)} samples) for a {nwin}-sample window")
    win = get_window("hann", nwin, fftbins=True)
    sft = ShortTimeFFT(win, hop=hop, fs=fs, fft_mode="onesided", scale_to="psd", mfft=nwin)
    S = sft.stft(x)  # (F, T) complex, frames centred on t = k*hop/fs starting at 0
    freqs = sft.f
    times = sft.t(len(x))
    keep = freqs <= fmax_hz
    power = (np.abs(S[keep, :]) ** 2).T  # (T, F)
    power_db = (10.0 * np.log10(power + 1e-20)).astype(np.float32)
    params = {
        "fs": int(fs),
        "window_s": window_s,
        "hop_s": hop_s,
        "nwin": nwin,
        "hop": hop,
        "fmax_hz": fmax_hz,
        "window": "hann",
        "scale": "psd",
        "whiten_f_window_hz": whiten_f_window_hz,
        "whiten_t_window_s": whiten_t_window_s,
    }
    if whiten:
        w = whiten_power_db(
            power_db,
            float(times[1] - times[0]) if len(times) > 1 else hop_s,
            float(freqs[1] - freqs[0]),
            whiten_f_window_hz,
            whiten_t_window_s,
        )
    else:
        w = power_db.copy()
    return Spectrogram(
        np.asarray(times, dtype=np.float64),
        np.asarray(freqs[keep], dtype=np.float64),
        power_db,
        w.astype(np.float32),
        params,
    )


def _block_median_background(
    a: NDArray[np.float32], block: int, kernel: int
) -> NDArray[np.float32]:
    """Median background along axis 1 computed on block-medians then linearly upsampled."""
    T, F = a.shape
    nb = int(np.ceil(F / block))
    pad = nb * block - F
    ap = np.pad(a, ((0, 0), (0, pad)), mode="edge") if pad else a
    blocks = np.median(ap.reshape(T, nb, block), axis=2)  # (T, nb)
    smooth = median_filter(blocks, size=(1, max(1, kernel)), mode="nearest")
    centres = (np.arange(nb) + 0.5) * block
    cols = np.arange(F)
    out = np.empty_like(a)
    for i in range(T):
        out[i] = np.interp(cols, centres, smooth[i])
    return out


def _block_median_time(
    a: NDArray[np.float32], hop_s: float, t_window_s: float
) -> NDArray[np.float32]:
    """Per-frequency-bin median over consecutive time blocks of ~t_window_s, interpolated."""
    T = a.shape[0]
    nblk = max(1, int(round(T * hop_s / t_window_s)))
    edges = np.linspace(0, T, nblk + 1).astype(int)
    meds = np.stack(
        [np.median(a[edges[i] : edges[i + 1]], axis=0) for i in range(nblk)]
    )  # (nblk, F)
    if nblk == 1:
        return np.broadcast_to(meds[0], a.shape).astype(np.float32)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows = np.arange(T)
    out = np.empty_like(a)
    for f in range(a.shape[1]):
        out[:, f] = np.interp(rows, centres, meds[:, f])
    return out


def whiten_power_db(
    power_db: NDArray[np.float32],
    hop_s: float,
    df_hz: float,
    f_window_hz: float = 30.0,
    t_window_s: float = 600.0,
) -> NDArray[np.float32]:
    """Robust whitening in dB: subtract frequency background and stationary lines, scale by MAD."""
    a = power_db.astype(np.float32)
    T, F = a.shape
    block = max(1, int(round(2.0 / df_hz)))  # ~2 Hz blocks
    kernel = max(3, int(round(f_window_hz / (block * df_hz))) | 1)
    bg = _block_median_background(a, block, kernel)
    r = a - bg
    # stationary lines: per-bin median over long time blocks (hum has constant frequency,
    # aircraft ridges sweep through a bin quickly). Block medians are interpolated in time.
    stationary = _block_median_time(r, hop_s, t_window_s)
    # only subtract where the stationary component is clearly positive (a persistent line)
    r = r - np.clip(stationary, 0.0, None)
    mad = np.median(np.abs(r - np.median(r))) + 1e-6
    sigma = 1.4826 * mad
    return (r / sigma).astype(np.float32)


def ridge_from_peaks(
    spec: Spectrogram, t0: float, t1: float, f_lo: float, f_hi: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-frame argmax of the whitened map inside a box: (times, freqs, whitened values).

    A simple helper for tests and quick looks; the real detector is in :mod:`ridges`.
    """
    ti = np.where((spec.times >= t0) & (spec.times <= t1))[0]
    fi = np.where((spec.freqs >= f_lo) & (spec.freqs <= f_hi))[0]
    sub = spec.whitened[np.ix_(ti, fi)]
    k = np.argmax(sub, axis=1)
    # parabolic interpolation around the peak for sub-bin precision
    fr = spec.freqs[fi]
    peak_f = fr[k].astype(np.float64)
    vals = sub[np.arange(len(ti)), k].astype(np.float64)
    for i, kk in enumerate(k):
        if 0 < kk < len(fi) - 1:
            y0, y1, y2 = sub[i, kk - 1], sub[i, kk], sub[i, kk + 1]
            denom = y0 - 2 * y1 + y2
            if denom < 0:
                delta = 0.5 * (y0 - y2) / denom
                peak_f[i] = fr[kk] + delta * spec.df_hz
    return spec.times[ti], peak_f, vals
