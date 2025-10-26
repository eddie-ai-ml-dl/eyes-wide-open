import numpy as np
from modules.tracker_logic import signed_distance_to_curve

# -------------------------
# Auto-orientation function
# -------------------------
def auto_orient_curve(curve, sample_points, eps=2.0, min_crossings=3):
    """
    Auto-detect curve orientation (+1 or -1) based on sample points crossing the curve.

    Parameters
    ----------
    curve : np.ndarray
        Nx2 array of curve points
    sample_points : list of (x, y)
        List of anchor points from tracked trajectories
    eps : float
        Minimum distance from curve to consider a point for crossing detection
    min_crossings : int
        Minimum number of valid crossings needed to determine orientation

    Returns
    -------
    orientation : int
        +1 or -1
    diagnostics : dict
        Detailed info for debugging
    """
    last_sign=None
    crossings=[]
    candidates={"+1": 0, "-1": 0}

    for idx, pt in enumerate(sample_points):
        dist_signed=signed_distance_to_curve(pt, curve)
        # Ignore points very close to the curve
        if abs(dist_signed)<eps:
            continue

        current_sign=np.sign(dist_signed)
        if last_sign is not None and current_sign!=last_sign:
            # Record crossing
            weighted_value=abs(dist_signed)
            crossings.append((len(crossings), idx, last_sign, current_sign, weighted_value))
            # Update candidate totals
            if current_sign>last_sign:
                candidates["+1"]+=weighted_value
            else:
                candidates["-1"]+=weighted_value

        last_sign=current_sign

    # Decide orientation
    if len(crossings)<min_crossings:
        # Not enough crossings to decide reliably, default +1
        orientation=1
    else:
        orientation=1 if candidates["+1"]>=candidates["-1"] else -1

    diagnostics={
        "num_samples": len(sample_points),
        "num_crossings": len(crossings),
        "crossings": crossings,
        "candidates": candidates,
        "orientation": orientation
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
