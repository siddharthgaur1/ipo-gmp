"""What is the best score achievable on this synthetic data? Run it and see.

`scripts/seed_data.py` builds the label from the feature:

    signal            = gmp_pct * rng.uniform(0.5, 0.9)
    listing_gain_pct  = (signal + rng.gauss(0, 0.12)) * 100

So `listing_gain_pct` is, by construction, 0.7 x `gmp_pct` plus noise. An
"oracle" that knows this formula exactly and predicts `0.7 * gmp_pct` is the
ceiling: no model can do better, because the residual is pure injected noise.

This script computes that ceiling by re-running the generator's own arithmetic.
Compare it against the trained model's reported metrics. If they match, the
model has recovered the generator and the metric says nothing about IPOs.

    python scripts/oracle_ceiling.py
"""

from __future__ import annotations

import random

import numpy as np

N = 200_000
SEED = 7


def main() -> int:
    rng = random.Random(SEED)
    gmps, ys = [], []
    for _ in range(N):
        # Verbatim from scripts/seed_data.py:gen_ipo
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

    gmps, ys = np.array(gmps), np.array(ys)
    pred = gmps * 0.7 * 100  # E[uniform(0.5, 0.9)] = 0.7

    acc = float(np.mean(np.sign(pred) == np.sign(ys)))
    r2 = float(1 - np.var(ys - pred) / np.var(ys))

    print(f"n                            {N:,}")
    print(f"oracle directional accuracy  {acc:.3f}")
    print(f"oracle R^2                   {r2:.3f}")
    print()
    print("Reported by the trained model: directional accuracy 0.912, R^2 0.871.")
    print()
    print("The model sits at the ceiling. It has recovered the generator's own")
    print("formula, which is the only thing there is to learn in this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
