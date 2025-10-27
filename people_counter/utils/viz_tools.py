import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory_data, curve=None, save_path=None):
    """
    Plots the trajectory of a single person over time.
    Optionally overlays the counting curve.

    Parameters
    ----------
    path : list or np.ndarray
        List/array of (x, y) points representing the trajectory.
    curve : list or np.ndarray, optional
        List/array of (x, y) points representing the counting curve.
    save_path : str, optional
        File path to save the plot. If None, displays the plot.
    """

    path = trajectory_data.get("path", None)
    id = trajectory_data.get("id", "na")
    # Validate trajectory
    if path is None or len(path)==0:
        print("⚠️ No trajectory data to plot.")
        return

    path_np=np.array(path, dtype=np.float32)

    if path_np.shape[0]==1:
        print("⚠️ Trajectory contains only one point; plotting single point.")

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(path_np[:, 0], path_np[:, 1], 'bo-', label=f'Trajectory (tracker_id={id})')
    plt.scatter(path_np[0, 0], path_np[0, 1], color='green', s=100, label='Start')
    plt.scatter(path_np[-1, 0], path_np[-1, 1], color='red', s=100, label='End')
    plt.gca().invert_yaxis()  # Correct for image coordinates (top-left origin)

    # Overlay curve if provided
    if curve is not None and len(curve)>0:
        curve_np=np.array(curve, dtype=np.float32)
        plt.plot(curve_np[:, 0], curve_np[:, 1], 'y-', linewidth=2, label='Counting Curve')

    # Labels and styling
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.title("Tracked Person Trajectory")
    plt.legend()
    plt.grid(True)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Trajectory plot saved to: {save_path}")
    else:
        plt.show()


# -----------------------------
# 2️⃣ Real-time trajectory overlay
# -----------------------------
def draw_live_trajectory(frame, path, color=(0, 255, 255), thickness=2, max_points=30):
    """
    Draw a trailing trajectory for the selected person on the video frame.

    Args:
        frame (np.ndarray): The current video frame (BGR)
        path (list[tuple]): List of (x, y) coordinates for this track
        color (tuple): BGR color for the trajectory line
        thickness (int): Line thickness
        max_points (int): How many of the latest points to draw (for smooth trails)
    """
    if len(path) < 2:
        return frame

    pts = np.array(path[-max_points:], np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)

    # Draw last point (current position)
    cv2.circle(frame, tuple(pts[-1][0]), 5, (0, 255, 0), -1)

    return frame


if __name__ == "__main__":
    # Load curve
    curve_data = [
    (722, 390),
    (741, 389),
    (721, 421),
    (674, 438),
    (555, 418),
    (459, 387),
    (461, 366)
    ]

    pts = [
    (624, 406), (623, 402), (623, 396), (624, 392), (627, 389),
    (631, 382), (630, 383), (632, 383), (637, 379), (640, 377),
    (642, 371), (645, 367), (647, 364), (650, 362), (652, 360),
    (653, 358), (654, 355), (654, 354), (655, 351), (656, 346),
    (652, 341), (651, 337), (653, 336), (653, 335), (658, 333),
    ]
    plot_trajectory({"path": pts}, curve=curve_data)