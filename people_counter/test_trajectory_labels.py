from polars.dependencies import numpy

from modules.curve_utils import load_curve_config
from modules.orientation import auto_orient_curve
from utils.helper_functions import label_trajectory
import numpy as np

CURVE_CONFIG_PATH = "config/curve_config.json"

curve_data = load_curve_config(CURVE_CONFIG_PATH)
curve_np = np.array(curve_data, dtype=np.float32)

trajectory = [
    (485, 370), (485, 366), (487, 368), (488, 367), (488, 367), (488, 366),
    (489, 367), (490, 364), (490, 364), (491, 367), (494, 363), (495, 359),
    (495, 357), (496, 357), (496, 356), (495, 356), (495, 355), (494, 355),
    (494, 357), (494, 358), (497, 359), (497, 361), (499, 361), (500, 359),
    (502, 363), (500, 370), (501, 373), (500, 376), (503, 374), (503, 376),
    (501, 376), (499, 377), (497, 378), (494, 377), (493, 377), (493, 378),
    (492, 378), (492, 378), (492, 380), (492, 382), (493, 384), (492, 385),
    (492, 388), (492, 390), (492, 391), (492, 392), (492, 392), (493, 392),
    (494, 392), (496, 392), (498, 391), (500, 391), (499, 389), (498, 387),
    (501, 388), (498, 390), (497, 391), (497, 391), (496, 392), (494, 392),
    (494, 392), (493, 391), (491, 391), (493, 390), (496, 385), (496, 384),
    (498, 382), (496, 384), (496, 386), (491, 393), (487, 393), (476, 394),
    (474, 396), (473, 396), (474, 398), (473, 400), (470, 402), (469, 405),
    (466, 406), (464, 406), (462, 406), (460, 406), (457, 406), (454, 405),
    (453, 405), (452, 404), (450, 403), (450, 400), (446, 402), (446, 395),
    (447, 388), (446, 395), (441, 408), (411, 419), (413, 420), (416, 420),
    (411, 419), (405, 418), (402, 417), (400, 417), (398, 417), (396, 419)]

orientation, diag = auto_orient_curve(curve_np, trajectory, eps=3.0, min_crossings=1)
print("Auto-orientation result:", orientation)
print("Diagnostics:", diag)

labels = label_trajectory(curve_np, trajectory, orientation)

for pt, label in zip(trajectory, labels):
    print(f"{pt} -> {label}")
