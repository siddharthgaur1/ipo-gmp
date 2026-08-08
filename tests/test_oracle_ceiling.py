"""Guard the circularity finding: the model must not appear to beat the ceiling.

scripts/seed_data.py builds the label from the feature, so an oracle predicting
0.7 * gmp_pct is optimal. If a future change to the generator alters that
relationship, the README's ceiling table becomes wrong -- and a README claiming
a model sits at a ceiling that has moved is worse than no README.

This asserts the generator still has the property the README describes.
"""

from __future__ import annotations

import random

import numpy as np


def _regenerate(n: int = 20_000, seed: int = 7):
    """Re-run the generator's arithmetic (see scripts/seed_data.py:gen_ipo)."""
    rng = random.Random(seed)
    gmps, ys = [], []
    for _ in range(n):
        qib = rng.uniform(0.1, 150)
        nii = rng.uniform(0.1, 300)
        ret = rng.uniform(0.2, 120)
        total = (qib + nii * 0.35 + ret * 0.35) / 1.7
        cat = "SME" if rng.random() < 0.4 else "Mainboard"
        base = (rng.gauss(0.20, 0.30) * min(total / 50, 3)
                + rng.gauss(0, 0.05)
                + (0.10 if cat == "SME" else 0))
        gmp = max(-0.50, min(1.80, base))
        y = max(-60.0, min(200.0,
                           (gmp * rng.uniform(0.5, 0.9) + rng.gauss(0, 0.12)) * 100))
        gmps.append(gmp)
        ys.append(y)
    return np.array(gmps), np.array(ys)


def test_the_label_is_still_built_from_the_feature():
    """If this fails the generator changed and the README must be rechecked."""
    gmps, ys = _regenerate()
    corr = float(np.corrcoef(gmps, ys)[0, 1])
    assert corr > 0.85, (
        f"label/feature correlation {corr:.3f} -- the generator no longer builds "
        "the label from gmp_pct, so the ceiling table in the README is stale"
    )


def test_the_oracle_ceiling_is_where_the_readme_says_it_is():
    gmps, ys = _regenerate()
    pred = gmps * 0.7 * 100

    acc = float(np.mean(np.sign(pred) == np.sign(ys)))
    r2 = float(1 - np.var(ys - pred) / np.var(ys))

    # README quotes 0.910 / 0.872 at n=200k; allow sampling slack at n=20k.
    assert 0.88 < acc < 0.94, f"directional ceiling moved to {acc:.3f}"
    assert 0.83 < r2 < 0.91, f"R2 ceiling moved to {r2:.3f}"


def test_the_model_does_not_beat_the_ceiling():
    """A reported score above the oracle would mean extra leakage, not skill."""
    gmps, ys = _regenerate()
    pred = gmps * 0.7 * 100
    ceiling = float(np.mean(np.sign(pred) == np.sign(ys)))

    reported = 0.912  # README's committed directional accuracy
    assert reported <= ceiling + 0.02, (
        f"model reports {reported} against a ceiling of {ceiling:.3f}; "
        "a score above the oracle means a second leak, not a better model"
    )
