import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_trajectory(path, curve=None, save_path=None):
    """
    Plots the collected trajectory of a single person over time.
    Optionally overlays the counting curve.
    """
    if not path:
        print("⚠️ No trajectory data to plot.")
        return

    path_np=np.array(path)

    plt.figure(figsize=(8, 6))
    plt.plot(path_np[:, 0], path_np[:, 1], 'bo-', label='Trajectory (Track ID)')
    plt.scatter(path_np[0, 0], path_np[0, 1], color='green', s=100, label='Start')
    plt.scatter(path_np[-1, 0], path_np[-1, 1], color='red', s=100, label='End')
    plt.gca().invert_yaxis()  # Correct for image coordinate origin (top-left)

    if curve is not None:
        curve_np=np.array(curve)
        plt.plot(curve_np[:, 0], curve_np[:, 1], 'y-', linewidth=2, label='Counting Curve')

    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.title("Tracked Person Trajectory")
    plt.legend()
    plt.grid(True)

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
    curve_data=curve_data = [
    (722, 390),
    (741, 389),
    (721, 421),
    (674, 438),
    (555, 418),
    (459, 387),
    (461, 366)
    ]

    pts = [
        (705, 468), (706, 466), (706, 460), (707, 457), (708, 452),
        (709, 447), (708, 446), (708, 445), (708, 444), (707, 439),
        (707, 438), (708, 437), (708, 434), (708, 432), (710, 407),
        (711, 396), (712, 415), (712, 422), (712, 424), (711, 426),
        (710, 425), (710, 423), (709, 421), (707, 420), (705, 418),
        (704, 416), (703, 414), (703, 414), (703, 413), (702, 410),
        (701, 407), (701, 389), (700, 374), (697, 366), (697, 363),
        (697, 389), (696, 397), (694, 399), (694, 398), (693, 399),
        (693, 399), (692, 375), (692, 365), (691, 360), (686, 356),
        (685, 354), (683, 356), (681, 359), (680, 358), (679, 362),
        (678, 372), (678, 378), (677, 381), (675, 392), (673, 399),
        (672, 389), (671, 411), (670, 407), (669, 416), (669, 416),
        (669, 418), (668, 420), (667, 418), (667, 416), (667, 414),
        (666, 410), (665, 414), (660, 413), (656, 408), (655, 406),
        (652, 406), (650, 407), (652, 393), (642, 400), (647, 387),
        (651, 380), (652, 382), (652, 381), (653, 380)
    ]
    plot_trajectory(pts, curve=curve_data)