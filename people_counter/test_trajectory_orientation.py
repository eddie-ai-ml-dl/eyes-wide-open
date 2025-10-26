from modules.tracker_logic import signed_distance_to_curve
from modules.curve_utils import load_curve_config
from modules.orientation import auto_orient_curve
import numpy as np

def test_trajectory_labels(curve, trajectory, eps=3.0, min_crossings=1, dist_thresh=50, sample_sz=10):
    """
    Labels points in a trajectory as IN or OUT with respect to the curve,
    using auto-orientation based on proximity-based sampling.
    """
    trajectory_np = np.array(trajectory, dtype=np.float32)

    # Compute distance of each point to the curve
    distances = np.array([abs(signed_distance_to_curve(pt, curve)) for pt in trajectory_np])

    # Select points within distance threshold
    close_points = trajectory_np[distances <= dist_thresh]

    if len(close_points) == 0:
        # Fallback: use first `sample_sz` points
        sample_anchors = trajectory_np[:min(sample_sz, len(trajectory_np))]
        print(f"No points within dist_thresh={dist_thresh}. Using first {len(sample_anchors)} points as anchors.")
    else:
        # If too many close points, randomly sample up to sample_sz
        if len(close_points) > sample_sz:
            indices = np.random.choice(len(close_points), sample_sz, replace=False)
            sample_anchors = close_points[indices]
        else:
            sample_anchors = close_points
        print(f"Selected {len(sample_anchors)} sample anchors based on proximity to curve:")

    for idx, pt in enumerate(sample_anchors):
        print(f"  Anchor {idx}: {pt}")

    # Determine orientation
    orientation, diagnostics = auto_orient_curve(curve, sample_anchors, eps=eps, min_crossings=min_crossings)
    print("Auto-orientation result:", orientation)
    print("Diagnostics:", diagnostics)

    # Label each point
    labeled_points = []
    for pt in trajectory:
        dist = signed_distance_to_curve(pt, curve)
        oriented_dist = dist * orientation
        label = "IN" if oriented_dist > 0 else "OUT"
        labeled_points.append((pt, dist, oriented_dist, label))
        print(f"{pt} -> dist={dist:.2f}, oriented={oriented_dist:.2f}, label={label}")

    return labeled_points


if __name__ == "__main__":
    CURVE_CONFIG_PATH="config/curve_config.json"

    # Load curve
    curve_data=load_curve_config(CURVE_CONFIG_PATH)
    curve=np.array(curve_data, dtype=np.float32)  # Assuming load_curve_config returns list of points

    # Example usage:
    # trajectory_example=[
    #     (485, 370), (485, 366), (487, 368), (488, 367), (488, 366),
    #     (489, 367), (490, 364), (491, 367), (494, 363), (495, 359),
    #     (500, 370), (476, 394), (474, 396), (473, 396), (470, 402),
    #     (466, 406), (462, 406), (450, 400)
    # ]

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

    trajectory_example= [
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

    # Run the test
    test_trajectory_labels(curve, trajectory_example)
