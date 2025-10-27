# modules/orientation.py
import numpy as np
from typing import Tuple, List, Dict
from modules.tracker_logic import signed_distance_to_curve


def auto_orient_curve(
    curve: np.ndarray,
    sample_points: List[tuple],
    eps: float = 3.0,
    min_crossings: int = 2,
    min_samples: int = 6,
    require_balance: bool = False
) -> Tuple[int, Dict]:
    """
    Auto-detect orientation (+1 or -1) from sample anchor points.

    Parameters
    ----------
    curve : np.ndarray
        Nx2 array of curve points.
    sample_points : list of (x,y)
        Candidate anchor points (can come from many tracks/frames).
    eps : float
        Ignore points whose abs(signed distance) < eps (too close to curve).
    min_crossings : int
        Minimum number of detected sign-change crossings required to trust automatic decision.
    min_samples : int
        Minimum number of sample_points to attempt auto-orientation; otherwise return default.
    require_balance : bool
        If True, require both directions to have some support to avoid one-direction bias.

    Returns
    -------
    orientation : int
        +1 or -1 (multiplier to apply to signed distances).
    diagnostics : dict
        { 'num_samples', 'num_used', 'num_crossings', 'crossings', 'candidates',
          'orientation', 'confidence', 'notes' }
    """
    diagnostics = {
        "num_samples": len(sample_points),
        "num_used": 0,
        "num_crossings": 0,
        "crossings": [],
        "candidates": {"+1": 0.0, "-1": 0.0},
        "orientation": 1,
        "confidence": 0.0,
        "notes": []
    }

    if len(sample_points) < min_samples:
        diagnostics["notes"].append(f"too_few_samples (need >= {min_samples})")
        return 1, diagnostics

    last_sign = None
    last_dist = None
    last_idx = None
    used_count = 0

    for idx, pt in enumerate(sample_points):
        d = float(signed_distance_to_curve(tuple(pt), curve))
        if abs(d) < eps:
            continue  # skip near-curve points
        cur_sign = 1 if d > 0 else -1

        if last_sign is not None and cur_sign != last_sign:
            weight = abs(d) + abs(last_dist) if last_dist is not None else abs(d)
            diagnostics["crossings"].append((last_idx, idx, last_sign, cur_sign, float(weight)))

            if cur_sign > last_sign:
                diagnostics["candidates"]["+1"] += float(weight)
            else:
                diagnostics["candidates"]["-1"] += float(weight)

            diagnostics["num_crossings"] += 1

        last_sign = cur_sign
        last_dist = d
        last_idx = idx
        used_count += 1

    diagnostics["num_used"] = used_count
    pos_votes = diagnostics["candidates"]["+1"]
    neg_votes = diagnostics["candidates"]["-1"]

    if diagnostics["num_crossings"] < min_crossings:
        diagnostics["notes"].append("not_enough_crossings")
        orientation = 1
    else:
        orientation = 1 if pos_votes >= neg_votes else -1

    total = pos_votes + neg_votes
    confidence = float((pos_votes if orientation == 1 else neg_votes) / total) if total > 0 else 0.0
    diagnostics["orientation"] = orientation
    diagnostics["confidence"] = confidence

    if require_balance and confidence > 0.95:
        diagnostics["notes"].append("high_confidence_but_check_for_morning_bias")

    return orientation, diagnostics

def auto_orient_curve001(curve, sample_points, eps=2.0, min_crossings=3):
    """
    Auto-detect curve orientation (+1 or -1) based on sample points crossing the curve.

    Camera assumption
    -----------------
    - The camera faces the building entrance.
    - People walking *toward* the camera are considered IN.
    - Positive oriented distance → toward camera → IN.
    - Negative oriented distance → away from camera → OUT.

    Parameters
    ----------
    curve : np.ndarray
        Nx2 array of curve points.
    sample_points : list of (x, y)
        List of anchor points from tracked trajectories.
    eps : float
        Minimum distance from curve to consider a point for crossing detection.
    min_crossings : int
        Minimum number of valid crossings needed to determine orientation.

    Returns
    -------
    orientation : int
        +1 or -1, adjusted for camera-facing-entrance convention.
    diagnostics : dict
        Detailed info for debugging.
    """

    last_sign = None
    crossings = []
    candidates = {"+1": 0, "-1": 0}

    for idx, pt in enumerate(sample_points):
        dist_signed = signed_distance_to_curve(pt, curve)

        # Ignore points very close to the curve
        if abs(dist_signed) < eps:
            continue

        current_sign = np.sign(dist_signed)

        if last_sign is not None and current_sign != last_sign:
            # Record crossing
            weighted_value = abs(dist_signed)
            crossings.append((len(crossings), idx, last_sign, current_sign, weighted_value))

            # Update candidate totals
            if current_sign > last_sign:
                candidates["+1"] += weighted_value
            else:
                candidates["-1"] += weighted_value

        last_sign = current_sign

    # Decide orientation
    if len(crossings) < min_crossings:
        orientation = 1  # Not enough crossings; default to +1
    else:
        orientation = 1 if candidates["+1"] >= candidates["-1"] else -1

    # ✅ Adjust for "camera facing entrance" convention if needed
    # (Flip this sign only if tests show reversed labeling)
    orientation *= 1  # change to -1 if IN/OUT appear inverted

    diagnostics = {
        "num_samples": len(sample_points),
        "num_crossings": len(crossings),
        "crossings": crossings,
        "candidates": candidates,
        "orientation": orientation,
        "camera_convention": "facing_entrance"
    }

    return orientation, diagnostics


def auto_orient_curve00(curve, sample_points, eps=3.0, min_crossings=1):
    """
    Robust orientation calibration.

    Args:
      curve: np.array of curve points shape (N,2)
      sample_points: list of (x,y) anchor points collected over time
      eps: hysteresis buffer (ignore |dist| < eps as 'on the curve')
      min_crossings: minimum number of true sign-flip crossings needed to decide;
                     if fewer crossings are found, function returns 1 (default).

    Returns:
      orientation: +1 or -1 (multiplier to apply to signed distances)
      diagnostics: dict with details (optional, helpful for debug)
    """
    def signed(pt):
        d = signed_distance_to_curve(pt, curve)
        if abs(d) < eps:
            return 0, d
        return (1 if d > 0 else -1), d

    # build sign sequence with indices and distances
    seq = []
    for idx, p in enumerate(sample_points):
        s, d = signed(p)
        seq.append((idx, p, s, d))

    # detect real crossings: transitions between non-zero opposite signs
    crossings = []
    last_idx = None
    last_sign = None
    for idx, p, s, d in seq:
        if s == 0:
            # skip samples too close to the curve
            continue
        if last_sign is None:
            last_sign = s
            last_idx = idx
            continue
        if s != last_sign:
            # we observed a crossing between last_idx and idx
            # compute dy = y_after - y_before (positive => moving down)
            y_before = sample_points[last_idx][1]
            y_after = p[1]
            dy = float(y_after) - float(y_before)
            crossings.append((last_idx, idx, last_sign, s, dy))
            last_sign = s
            last_idx = idx

    diagnostics = {
        "num_samples": len(sample_points),
        "num_crossings": len(crossings),
        "crossings": crossings[:10],  # preview
    }

    # If not enough crossings, fallback to default (1)
    if len(crossings) < min_crossings:
        diagnostics["reason"] = "not_enough_crossings; fallback to +1"
        return 1, diagnostics

    # For each crossing compute candidate orientation:
    # - if dy > 0 (moving down) then candidate = old_sign
    #   (because multiplying by old_sign will make the transition old*o -> new*o be positive->negative)
    # - if dy < 0 (moving up) then candidate = -old_sign
    candidates = []
    for last_idx, idx, old_sign, new_sign, dy in crossings:
        cand = old_sign if dy > 0 else -old_sign
        candidates.append(cand)

    # majority vote
    ones = sum(1 for c in candidates if c == 1)
    negs = sum(1 for c in candidates if c == -1)
    orientation = 1 if ones >= negs else -1

    diagnostics["candidates"] = {"+1": ones, "-1": negs}
    diagnostics["orientation"] = orientation
    return orientation, diagnostics
