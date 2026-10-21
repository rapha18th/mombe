"""Calibrated scale claim.

The reconstruction's own scale error (jackknife within each board placement,
spread between placements) under-states the true error. On the fourteen body
sweep takes (seven synthetic animals on the stock and COOL builds; input S,
truth G) the raw claim covered the true error on 9 at one sigma and on 11
inside a 90% band. The claim the gates and the weight interval read is

    sigma = sqrt(FLOOR^2 + (MULT * claimed)^2)

FLOOR covers takes whose jackknife reads near zero while the scale is still off
by a few tenths of a percent (stock sweep_27: claimed 0.06%, true -0.46%); MULT
covers takes whose claim is right in size and too small (COOL sweep_21: claimed
4.4%, true -8.9%). eval/body_gates.py fits both as the narrowest pair whose 90%
band covers every take. Held out, the band covered 12 of 14 takes (each take
left out) and 11 of 14 (each animal left out on both builds); seven animals
make that a small sample. Pure Python, so the policy side can import it
without OpenCV.
"""

from __future__ import annotations

import math
import os

Z90 = 1.645
SCALE_CLAIM_FLOOR = float(os.environ.get("MOMBE_SCALE_CLAIM_FLOOR", 0.0032))
SCALE_CLAIM_MULT = float(os.environ.get("MOMBE_SCALE_CLAIM_MULT", 1.25))


def scale_claim(claimed: float) -> float:
    """The calibrated relative scale error (one sigma) for a claimed one."""
    return math.sqrt(SCALE_CLAIM_FLOOR ** 2 + (SCALE_CLAIM_MULT * claimed) ** 2)
