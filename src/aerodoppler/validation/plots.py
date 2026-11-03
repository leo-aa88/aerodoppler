"""Diagnostic plots (PNG) for reports and the demo: spectrograms with ridge/curve overlays,
candidate maps and score landscapes. Headless matplotlib only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aerodoppler.audio.spectrogram import Spectrogram  # noqa: E402


def plot_spectrogram(
    spec: Spectrogram,
    out: Path,
    title: str = "",
    fmax: float | None = None,
    overlays: list[dict[str, Any]] | None = None,
    vmin: float = -2,
    vmax: float = 8,
    synthetic: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        spec.whitened.T,
        origin="lower",
        aspect="auto",
        extent=[spec.times[0], spec.times[-1], spec.freqs[0], spec.freqs[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    for ov in overlays or []:
        ax.plot(
            ov["t"],
            ov["f"],
            ov.get("style", "-"),
            lw=ov.get("lw", 1.0),
            color=ov.get("color", "cyan"),
            alpha=ov.get("alpha", 0.9),
            label=ov.get("label"),
        )
    ax.set_ylim(spec.freqs[0], fmax or spec.freqs[-1])
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(("SYNTHETIC DEMO — " if synthetic else "") + title)
    if overlays:
        ax.legend(fontsize=7, loc="upper right", ncol=3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_candidate_map(
    out: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    score: np.ndarray,
    tracks: list[dict[str, Any]],
    best: tuple[float, float] | None,
    truth: tuple[float, float] | None,
    title: str,
    synthetic: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 8))
    if len(lat):
        sc = ax.scatter(lon, lat, c=score, s=6, cmap="viridis", alpha=0.85)
        fig.colorbar(sc, ax=ax, label="relative fit score (uncalibrated)")
    for tr in tracks:
        ax.plot(tr["lon"], tr["lat"], "-", lw=1.2, alpha=0.8, label=tr.get("label"))
    if best is not None:
        ax.plot(best[1], best[0], "r*", ms=16, label="predicted receiver")
    if truth is not None:
        ax.plot(truth[1], truth[0], "wx", ms=12, mew=2.5, label="true receiver (synthetic)")
        ax.plot(truth[1], truth[0], "kx", ms=12, mew=1.0)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(("SYNTHETIC DEMO — " if synthetic else "") + title)
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lat) if len(lat) else 0.0)))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
