import numpy as np
import matplotlib.pyplot as plt
from modules.tracker_logic import signed_distance_to_curve
from modules.orientation import auto_orient_curve

def test_trajectory_labels(curve, trajectory, eps=3.0, min_crossings=1, sample_sz=10, IN_direction=None, plot=True):
    """
    Labels trajectory points as IN or OUT based on curve crossings and orientation.
    Flips label each time the trajectory crosses the curve.

    Parameters
    ----------
    curve : np.ndarray
        Nx2 array of curve points
    trajectory : list of (x, y)
        Trajectory points
    eps : float
        Minimum distance from curve to ignore small fluctuations
    min_crossings : int
        Minimum crossings to consider auto-orientation reliable
    sample_sz : int
        Number of initial points to sample for auto-orientation
    IN_direction : str
        Optional, one of 'toward_cam', 'away_from_cam', 'left', 'right'
    plot : bool
        Whether to plot trajectory with labels

    Returns
    -------
    labeled_points : list of tuples
        Each tuple: (point, signed_distance, oriented_distance, label)
    """
    trajectory_np = np.array(trajectory, dtype=np.float32)

    # Determine orientation
    if IN_direction is None or IN_direction == 'auto':
        sample_anchors = trajectory_np[:min(sample_sz, len(trajectory_np))]
        orientation, diagnostics = auto_orient_curve(curve, sample_anchors, eps=eps, min_crossings=min_crossings)
        print("Using orientation (auto-detected):", orientation)
        print("Diagnostics:", diagnostics)
    else:
        orientation = 1 if IN_direction in ["toward_cam", "left", "right"] else -1
        print(f"Using orientation from config IN_direction='{IN_direction}':", orientation)

    # Initialize labeling
    labeled_points = []

    first_oriented_dist = signed_distance_to_curve(trajectory_np[0], curve) * orientation
    prev_oriented_dist = first_oriented_dist

    # Set initial label
    current_label = "IN" if first_oriented_dist > 0 else "OUT"

    for pt in trajectory_np:
        dist = signed_distance_to_curve(pt, curve)
        oriented_dist = dist * orientation

        # Detect crossing (sign change beyond eps)
        if prev_oriented_dist * oriented_dist < 0 and abs(oriented_dist) > eps:
            # Flip label
            current_label = "IN" if current_label == "OUT" else "OUT"

        labeled_points.append((pt, dist, oriented_dist, current_label))
        prev_oriented_dist = oriented_dist

        print(f"{pt} -> dist={dist:.2f}, oriented={oriented_dist:.2f}, label={current_label}")

    if plot:
        plot_labeled_trajectory(trajectory_np, curve, labeled_points)

    return labeled_points

def plot_labeled_trajectory(trajectory_np, curve, labeled_points, save_path=None):
    """
    Plots trajectory with points colored by IN/OUT status.
    """
    plt.figure(figsize=(8, 6))

    # Plot curve
    if curve is not None:
        curve_np = np.array(curve, dtype=np.float32)
        plt.plot(curve_np[:, 0], curve_np[:, 1], 'y-', linewidth=2, label='Counting Curve')

    # Plot trajectory with colors
    for idx, (pt, _, _, label) in enumerate(labeled_points):
        color = 'green' if label == 'IN' else 'red'
        plt.scatter(pt[0], pt[1], color=color, s=50)
        if idx > 0:
            plt.plot([labeled_points[idx-1][0][0], pt[0]],
                     [labeled_points[idx-1][0][1], pt[1]],
                     color=color, linewidth=2)

    # Mark start and end
    plt.scatter(trajectory_np[0, 0], trajectory_np[0, 1], color='blue', s=100, label='Start')
    plt.scatter(trajectory_np[-1, 0], trajectory_np[-1, 1], color='black', s=100, label='End')

    plt.gca().invert_yaxis()
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.title("Trajectory with IN/OUT Labeling")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from utils.curve_utils import load_curve_config
    from modules.orientation import auto_orient_curve
    import numpy as np

    CURVE_CONFIG_PATH = "config/curve_config.json"
    curve_data = load_curve_config(CURVE_CONFIG_PATH)
    curve_points = np.array(curve_data["curve_points"], dtype=np.float32)
    IN_direction = curve_data.get("IN_direction", "auto")

    print(f"Using curve with IN_direction: {IN_direction}")

    trajectory_example = [
        (624, 406), (623, 402), (623, 396), (624, 392), (627, 389),
        (631, 382), (630, 383), (632, 383), (637, 379), (640, 377),
        (642, 371), (645, 367), (647, 364), (650, 362), (652, 360),
        (653, 358), (654, 355), (654, 354), (655, 351), (656, 346),
        (652, 341), (651, 337), (653, 336), (653, 335), (658, 333),
    ]

    trajectory_np = np.array(trajectory_example, dtype=np.float32)
    sample_sz = 10
    eps = 3.0
    min_crossings = 1

    if IN_direction == "auto":
        sample_anchors = trajectory_np[:min(sample_sz, len(trajectory_np))]
        orientation, diagnostics = auto_orient_curve(curve_points, sample_anchors, eps=eps, min_crossings=min_crossings)
        print(f"Using orientation (auto-detected): {orientation}")
        print("Diagnostics:", diagnostics)
    else:
        orientation = 1 if IN_direction in ["toward_cam", "left", "right"] else -1
        print(f"Using orientation from config IN_direction='{IN_direction}': {orientation}")

    # Run test
    labeled_points = test_trajectory_labels(
        curve=curve_points,
        trajectory=trajectory_example,
        sample_sz=10,
        IN_direction=IN_direction
    )