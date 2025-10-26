from modules.tracker_logic import signed_distance_to_curve

def label_trajectory(curve, trajectory, orientation):
    """
    Labels each point of a trajectory as 'IN' or 'OUT' based on the curve orientation.

    Parameters:
        curve (list of [x, y]): The curve points.
        trajectory (list of (x, y)): The trajectory points.
        orientation (int): Auto-orientation of the curve (+1 or -1).

    Returns:
        list of str: 'IN' or 'OUT' for each trajectory point.
    """
    labels = []
    for pt in trajectory:
        dist = signed_distance_to_curve(pt, curve)
        oriented_dist = dist * orientation
        labels.append("IN" if oriented_dist < 0 else "OUT")
    return labels
