"""Ranking, flags and abstention (SPEC §10.5): relative fit scores only, explicit abstention
reasons, ambiguity reporting when distinct locations score nearly as well."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerodoppler.models import Abstention


@dataclass
class RankConfig:
    min_score_z: float = 1.2  # mean whitened z along curves below this = no convincing ridge
    min_coverage: float = 0.25
    ambiguity_margin_z: float = 0.3
    min_candidates_for_ranking: int = 1


def decide_abstention(
    n_segments: int,
    n_assignments: int,
    best_score: float | None,
    best_coverage: float | None,
    cfg: RankConfig,
    event_flags: list[str],
) -> Abstention:
    if n_segments == 0:
        return Abstention(
            abstained=True,
            reason="no_flight_coverage",
            detail="no airborne flight segments with altitude overlap the date/time window and bbox (archive coverage gap, wrong date, or bbox)",
        )
    if n_assignments == 0:
        return Abstention(
            abstained=True,
            reason="no_candidate_geometry",
            detail="no grid point has flights whose delayed-CPA timing matches the event separation / known time within tolerance and the speed & slant bounds",
        )
    if best_score is None or not np.isfinite(best_score):
        return Abstention(
            abstained=True, reason="scoring_failed", detail="curve scoring produced no finite value"
        )
    if best_score < cfg.min_score_z:
        return Abstention(
            abstained=True,
            reason="weak_fit",
            detail=f"best relative fit score {best_score:.2f} < {cfg.min_score_z} (no convincing ridge energy along any predicted curve)",
        )
    if best_coverage is not None and best_coverage < cfg.min_coverage:
        return Abstention(
            abstained=True,
            reason="poor_coverage",
            detail=f"only {best_coverage:.0%} of the predicted curve shows ridge energy",
        )
    return Abstention(abstained=False)


def ambiguity_flags(scores_distinct: list[float], cfg: RankConfig) -> list[str]:
    if (
        len(scores_distinct) >= 2
        and scores_distinct[0] - scores_distinct[1] < cfg.ambiguity_margin_z
    ):
        return [
            f"ambiguous: runner-up distinct location within {cfg.ambiguity_margin_z} z of the best"
        ]
    return []
