import numpy as np
import matplotlib.pyplot as plt
from modules.tracker_logic import signed_distance_to_curve
from modules.orientation import auto_orient_curve

def test_trajectory_labels(curve, trajectory, eps=3.0, min_crossings=1, sample_sz=10, plot=True, IN_direction=None):
    """
    Labels points in a trajectory as IN or OUT with respect to the curve.

    Parameters
    ----------
    curve : np.ndarray
        Nx2 array of curve points
    trajectory : list of (x, y)
        Trajectory points
    eps : float
        Minimum distance for considering crossings
    min_crossings : int
        Minimum crossings to decide orientation if auto_orient used
    sample_sz : int
        Number of points used for auto_orientation
    plot : bool
        Whether to plot trajectory and curve
    IN_direction : str
        Optional: "toward_cam", "away_from_cam", "left", "right", etc.
        If provided, auto-orient is skipped.
    """
    trajectory_np = np.array(trajectory, dtype=np.float32)

    # Determine orientation
    if IN_direction is not None:
        orientation = 1  # always +1, distances will be interpreted according to IN_direction
        source = f"config IN_direction={IN_direction}"
        diagnostics = {"num_samples": len(trajectory_np), "num_crossings": 0, "orientation": orientation, "camera_convention": IN_direction}
    else:
        # Select sample points for orientation
        sample_anchors = trajectory_np[:min(sample_sz, len(trajectory_np))]
        orientation, diagnostics = auto_orient_curve(curve, sample_anchors, eps=eps, min_crossings=min_crossings)
        source = "auto_orient"

    print(f"Using orientation: {orientation} (source: {source})")
    print("Diagnostics:", diagnostics)

    # Labeling
    labeled_points = []
    for pt in trajectory_np:
        dist = signed_distance_to_curve(pt, curve)
        oriented_dist = dist * orientation

        # Label according to IN_direction / oriented distance
        if IN_direction is not None:
            if IN_direction in ["toward_cam", "left", "right"]:
                label = "IN" if oriented_dist > 0 else "OUT"
            elif IN_direction in ["away_from_cam"]:
                label = "IN" if oriented_dist < 0 else "OUT"
            else:
                label = "IN" if oriented_dist > 0 else "OUT"  # fallback
        else:
            # fallback to auto-crossing logic
            label = "IN" if oriented_dist > 0 else "OUT"

        labeled_points.append((pt, dist, oriented_dist, label))
        print(f"{pt} -> dist={dist:.2f}, oriented={oriented_dist:.2f}, label={label}")

    # Optional plotting
    if plot:
        plot_labeled_trajectory(trajectory_np, curve, labeled_points)

    return labeled_points

def plot_labeled_trajectory(trajectory_np, curve, labeled_points, save_path=None):
    plt.figure(figsize=(8, 6))

    # Plot curve
    if curve is not None:
        curve_np = np.array(curve, dtype=np.float32)
        plt.plot(curve_np[:, 0], curve_np[:, 1], 'y-', linewidth=2, label='Counting Curve')

    # Plot trajectory with colors
    for idx, (pt, dist, oriented_dist, label) in enumerate(labeled_points):
        color = 'green' if label == 'IN' else 'red'
        plt.scatter(pt[0], pt[1], color=color, s=50)
        if idx > 0:
            plt.plot([labeled_points[idx-1][0][0], pt[0]],
                     [labeled_points[idx-1][0][1], pt[1]],
                     color=color, linewidth=2)

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

    CURVE_CONFIG_PATH = "config/curve_config.json"

    # Load curve and IN_direction from config
    curve_data = load_curve_config(CURVE_CONFIG_PATH)
    # print(curve_data)
    curve_points = np.array(curve_data["curve_points"], dtype=np.float32)
    in_direction = curve_data.get("IN_direction", "toward_cam")  # default fallback

    print(f"Using curve with IN_direction: {in_direction}")

    # Example trajectory
    trajectory_example = [
        (624, 406), (623, 402), (623, 396), (624, 392), (627, 389),
        (631, 382), (630, 383), (632, 383), (637, 379), (640, 377),
        (642, 371), (645, 367), (647, 364), (650, 362), (652, 360),
        (653, 358), (654, 355), (654, 354), (655, 351), (656, 346),
        (652, 341), (651, 337), (653, 336), (653, 335), (658, 333),
    ]

    trajectory_example=[
        (485, 370), (485, 366), (487, 368), (488, 367), (488, 366),
        (489, 367), (490, 364), (491, 367), (494, 363), (495, 359),
        (500, 370), (476, 394), (474, 396), (473, 396), (470, 402),
        (466, 406), (462, 406),
    ]

    # direction OUT
    trajectory_example=[
            (392,438),(397,438),(401,437),(409,436),(421,436),
            (434,435),(446,434),(443,434),(454,434),(464, 433),
            (473, 430),(482, 429), (502, 426),(514, 423),
            (520, 420),(528, 417),(536, 416),(540, 414),
            (545, 413),(554, 414),(564, 414),(573, 413),
            (580, 413),(585, 414),(589, 414),(593, 413),
            (597, 413),(601, 412),(605, 409),(608, 407),
            (614, 408)
        ]

    # Run test
    labeled_points = test_trajectory_labels(
        curve=curve_points,
        trajectory=trajectory_example,
        sample_sz=10,
        IN_direction=in_direction
    )

    # Labeled points already printed and plotted

    # direction OUT
    # trajectory_example=[
    #         (392,438),(397,438),(401,437),(409,436),(421,436),
    #         (434,435),(446,434),(443,434),(454,434),(464, 433),
    #         (473, 430),(482, 429), (502, 426),(514, 423),
    #         (520, 420),(528, 417),(536, 416),(540, 414),
    #         (545, 413),(554, 414),(564, 414),(573, 413),
    #         (580, 413),(585, 414),(589, 414),(593, 413),
    #         (597, 413),(601, 412),(605, 409),(608, 407),
    #         (614, 408)
    #     ]

    # all OUTSIDE
    # trajectory_example= [
    # (586, 212), (588, 215), (589, 217), (590, 218), (592, 218),
    # (592, 217), (594, 218), (594, 218), (595, 218), (596, 219),
    # (596, 219), (597, 221), (598, 226), (598, 233), (598, 233),
    # (598, 232), (597, 231), (597, 231), (597, 230), (597, 229),
    # (597, 228), (597, 230), (597, 233), (598, 234), (599, 235),
    # (601, 237), (601, 231), (604, 227), (604, 227), (606, 227),
    # (605, 236), (604, 245), (603, 248), (602, 250), (600, 251),
    # (599, 248), (596, 247), (595, 248), (595, 248), (592, 250),
    # (591, 253), (591, 255), (592, 257), (595, 258), (595, 258),
    # (594, 259), (592, 262), (589, 263), (587, 265), (584, 267),
    # (582, 269), (579, 271), (578, 272), (575, 273), (572, 274),
    # (570, 276), (569, 278), (568, 281), (569, 285), (569, 287),
    # (569, 288), (569, 289), (569, 290), (568, 292), (568, 294),
    # (567, 297), (566, 300), (564, 302), (561, 310), (558, 312),
    # (552, 317), (549, 315), (546, 315), (544, 319), (543, 322),
    # (543, 322), (542, 322), (541, 323), (540, 324), (540, 325),
    # (541, 326), (541, 328), (539, 330), (538, 334), (536, 338),
    # (534, 342), (533, 342), (531, 343), (528, 344), (524, 343)
    # ]

    # back and forth
    # trajectory_example= [
    #     (705, 468), (706, 466), (706, 460), (707, 457), (708, 452),
    #     (709, 447), (708, 446), (708, 445), (708, 444), (707, 439),
    #     (707, 438), (708, 437), (708, 434), (708, 432), (710, 407),
    #     (711, 396), (712, 415), (712, 422), (712, 424), (711, 426),
    #     (710, 425), (710, 423), (709, 421), (707, 420), (705, 418),
    #     (704, 416), (703, 414), (703, 414), (703, 413), (702, 410),
    #     (701, 407), (701, 389), (700, 374), (697, 366), (697, 363),
    #     (697, 389), (696, 397), (694, 399), (694, 398), (693, 399),
    #     (693, 399), (692, 375), (692, 365), (691, 360), (686, 356),
    #     (685, 354), (683, 356), (681, 359), (680, 358), (679, 362),
    #     (678, 372), (678, 378), (677, 381), (675, 392), (673, 399),
    #     (672, 389), (671, 411), (670, 407), (669, 416), (669, 416),
    #     (669, 418), (668, 420), (667, 418), (667, 416), (667, 414),
    #     (666, 410), (665, 414), (660, 413), (656, 408), (655, 406),
    #     (652, 406), (650, 407), (652, 393), (642, 400), (647, 387),
    #     (651, 380), (652, 382), (652, 381), (653, 380)
    #
    # ]

    # trajectory_example= [
    # (624, 406), (623, 402), (623, 396), (624, 392), (627, 389),
    # (631, 382), (630, 383), (632, 383), (637, 379), (640, 377),
    # (642, 371), (645, 367), (647, 364), (650, 362), (652, 360),
    # (653, 358), (654, 355), (654, 354), (655, 351), (656, 346),
    # (652, 341), (651, 337), (653, 336), (653, 335), (658, 333),
    # ]

