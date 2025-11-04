import os
import json
import cv2
import numpy as np
import math
from modules.orientation import auto_orient_curve


class CurveManager:
    """
    Handles loading, creation, visualization, and saving of curve configurations
    for entrance/exit counting systems, with structured orientation diagnostics.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.curve_data = None

    # -------------------------
    # CONFIG IO
    # -------------------------
    def load_curve_config(self):
        """Load curve config (points + IN_direction + diagnostics) from JSON."""
        if not os.path.exists(self.config_path):
            print(f"⚠️ Curve config not found at {self.config_path}")
            return None
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
            self.curve_data = data
            print(f"✅ Loaded curve config from {self.config_path}")
            return data
        except Exception as e:
            print(f"❌ Failed to load curve config: {e}")
            return None

    def save_curve_config(self, data: dict):
        """Save curve configuration to disk."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Curve config saved to {self.config_path}")

    # -------------------------
    # helpers: rounding & numeric conversion
    # -------------------------
    @staticmethod
    def _round_floats(obj, decimals=2):
        """
        Recursively round floats inside nested structures (lists/dicts).
        Returns a new object (doesn't mutate input).
        """
        if isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        elif isinstance(obj, list):
            return [CurveManager._round_floats(v, decimals) for v in obj]
        elif isinstance(obj, dict):
            return {k: CurveManager._round_floats(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return CurveManager._round_floats(obj.tolist(), decimals)
        else:
            return obj

    # -------------------------
    # INTERACTIVE CREATION
    # -------------------------
    def create_curve(self, frame):
        """
        Interactive tool to draw a curve on a given frame.
        - Left-click: add points
        - Right-click or 'u': undo last point
        - 's' or double-click: save curve
        - ESC: cancel
        Returns dict: {"curve_points": [...], "IN_direction": "auto"}
        """
        print("\n🖱️  Click to add curve points")
        print("➡️  Controls: Left-click=add | Right-click/u=undo | s/double-click=save | ESC=cancel")

        clone = frame.copy()
        points = []
        done = False

        # Overlay helper text
        def draw_overlay(img):
            overlay = img.copy()
            instructions = "Left-click: add | Right-click/u: undo | s/dbl-click: save | ESC: cancel"
            cv2.putText(
                overlay, instructions, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                overlay, "Curve Definition Mode", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA
            )
            return overlay

        def draw_curve():
            disp = clone.copy()
            for i, pt in enumerate(points):
                cv2.circle(disp, pt, 4, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(disp, points[i - 1], pt, (0, 255, 0), 2)
            return draw_overlay(disp)

        def draw_callback(event, x, y, flags, param):
            nonlocal done
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.imshow("Define Curve", draw_curve())
            elif event == cv2.EVENT_RBUTTONDOWN:
                if points:
                    points.pop()
                    print("↩️  Undo last point (right-click)")
                    cv2.imshow("Define Curve", draw_curve())
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                done = True
                print("✅ Curve confirmed via double-click")

        cv2.namedWindow("Define Curve")
        cv2.setMouseCallback("Define Curve", draw_callback)
        cv2.imshow("Define Curve", draw_overlay(clone))

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("❌ Curve creation cancelled.")
                cv2.destroyWindow("Define Curve")
                return None
            elif key in [ord("u"), ord("U")]:
                if points:
                    points.pop()
                    print("↩️  Undo last point (keyboard)")
                    cv2.imshow("Define Curve", draw_curve())
            elif key in [ord("s"), ord("S")]:
                done = True
                print("✅ Curve confirmed via key 's'")
            if done:
                break

        cv2.destroyWindow("Define Curve")

        if len(points) < 2:
            print("❌ Not enough points selected.")
            return None

        curve_points = np.array(points, dtype=np.float32)

        data = {
            "curve_points": curve_points.tolist(),
            "IN_direction": "auto",
            "orientation": None,
            "camera_orientation": None,
            "orientation_diagnostics": {}
        }
        self.curve_data = data

        # Save to file immediately
        print("💾 Saving curve configuration...")
        self.save_curve_config(data)

        return data

    # -------------------------
    # ORIENTATION LOGIC
    # -------------------------
    def determine_orientation(self, sample_anchors):
        """
        Determine orientation based on either config or auto-detection.
        Updates and persists the config with full diagnostics.
        Resolves IN_direction to a concrete value.
        If we newly resolved orientation, recompute & persist inside region.
        """
        if not self.curve_data:
            raise ValueError("Curve must be loaded or created before determining orientation.")

        curve_np = np.array(self.curve_data["curve_points"], dtype=np.float32)
        in_dir = self.curve_data.get("IN_direction", "auto")

        diag = None
        if in_dir is None or in_dir == "auto":
            # Auto-detect orientation
            orientation, diag = auto_orient_curve(curve_np, sample_anchors, eps=3.0, min_crossings=1)
            print(f"🧭 Auto-detected curve orientation: {orientation}")
            print("Diagnostics:", diag)

            # Save orientation and diagnostics
            self.curve_data["orientation"] = int(orientation)
            self.curve_data["camera_orientation"] = diag.get("camera_convention", None)
            self.curve_data["orientation_diagnostics"] = diag

            # --- Resolve IN_direction based on orientation ---
            resolved_in_dir = "toward_cam" if orientation == 1 else "away_from_cam"
            self.curve_data["IN_direction"] = resolved_in_dir

            # Save back to config
            # Also (try) build inside region now that orientation is known
            try:
                frame_shape = diag.get("frame_shape", None)  # optional hint
                # If we have stored region_depth in config keep it, else default to 50
                region_depth = self.curve_data.get("region_depth", 50.0)
                inside_region = self.build_inside_region(
                    np.array(self.curve_data["curve_points"], dtype=np.float32),
                    resolved_in_dir,
                    frame_shape=frame_shape,
                    region_depth=float(region_depth),
                    orientation=int(orientation),
                )
                # Save inside region and diagnostics
                region_diag = {
                    "method": "normal_offset",
                    "region_depth": float(region_depth),
                    "num_points": int(len(inside_region)),
                    "status": "ok"
                }
                self.save_inside_region(inside_region, region_depth, region_diag)
            except Exception as e:
                print(f"❌ Failed to auto-build inside_region after orientation: {e}")

            self.save_curve_config(self.curve_data)
        else:
            # Already set in config
            orientation = 1 if in_dir in ["toward_cam", "left", "right"] else -1
            resolved_in_dir = in_dir
            print(f"➡️ Using orientation from config IN_direction='{in_dir}': {orientation}")

        return orientation, diag, resolved_in_dir

    # -------------------------
    # REGION GEOMETRY (NEW)
    # -------------------------
    @staticmethod
    def _compute_tangents_normals(curve_pts):
        """
        Compute tangent and normal (unit) vectors at each curve point.
        curve_pts: (N,2) ndarray
        Returns normals: (N,2) ndarray where normal points 'left' of tangent (i.e. rotate tangent by -90).
        """
        pts = np.asarray(curve_pts, dtype=np.float32)
        n = len(pts)
        tangents = np.zeros_like(pts)
        for i in range(n):
            if i == 0:
                tang = pts[1] - pts[0]
            elif i == n - 1:
                tang = pts[-1] - pts[-2]
            else:
                tang = (pts[i + 1] - pts[i - 1]) * 0.5
            norm = np.linalg.norm(tang)
            if norm < 1e-6:
                tangents[i] = np.array([1.0, 0.0])
            else:
                tangents[i] = tang / norm

        # normals: rotate tangent by -90° to get a consistent 'left' normal: (tx,ty) -> (-ty, tx)
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        # Normalize normals (safeguard)
        norms = np.linalg.norm(normals, axis=1)
        norms[norms < 1e-6] = 1.0
        normals = (normals.T / norms).T

        return tangents, normals

    @staticmethod
    def _offset_curve(curve_pts, normals, offset):
        """
        Offset curve points by normals * offset (offset may be + or -).
        """
        return curve_pts + (normals * offset)  # broadcasting

    @staticmethod
    def build_inside_region(curve_points, in_direction, frame_shape=None,
                            region_depth=50.0, orientation=None):
        """
        Construct INSIDE polygon region based on curve normals (normal-offset).
        - curve_points: Nx2 array-like
        - in_direction: string (e.g., 'toward_cam', 'away_from_cam', 'left','right','auto')
        - frame_shape: optional (h,w) or dict with 'h','w' values (used only if needed)
        - region_depth: offset distance in pixels
        - orientation: optional int (+1/-1) to specify which side is IN. If omitted and
                       in_direction == 'auto' a symmetric neutral region will be made.
        Returns: inside_region ndarray (M,2) float32 polygon points in order.
        """
        curve = np.asarray(curve_points, dtype=np.float32)
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError("curve_points must be Nx2 array-like")

        tangents, normals = CurveManager._compute_tangents_normals(curve)

        # Decide side: +1 means offset along normal, -1 means offset opposite normal.
        side_sign = None
        if orientation is not None:
            side_sign = 1 if int(orientation) == 1 else -1
        else:
            # try map from in_direction string if possible
            if in_direction in ["toward_cam"]:
                side_sign = 1
            elif in_direction in ["away_from_cam"]:
                side_sign = -1
            elif in_direction in ["left"]:
                side_sign = 1
            elif in_direction in ["right"]:
                side_sign = -1
            elif in_direction == "auto":
                side_sign = None
            else:
                side_sign = None

        # If orientation unknown / auto -> build neutral symmetric region (offset both sides)
        if side_sign is None:
            offs_pos = CurveManager._offset_curve(curve, normals, +region_depth)
            offs_neg = CurveManager._offset_curve(curve, normals, -region_depth)
            # polygon: pos_offset curve, then reversed neg_offset curve
            poly = np.vstack((offs_pos, offs_neg[::-1]))
        else:
            # Build one-sided region by offsetting curve inwards and joining original curve
            # Choose offset curve on the IN side and join curve + reversed offset side to make polygon
            offs_in = CurveManager._offset_curve(curve, normals, side_sign * region_depth)
            # Build polygon: original curve followed by reversed offset curve
            poly = np.vstack((curve, offs_in[::-1]))

        # Clip to frame bounds if provided
        if frame_shape is not None:
            try:
                if isinstance(frame_shape, dict):
                    h = int(frame_shape.get("h", frame_shape.get("height", 0)))
                    w = int(frame_shape.get("w", frame_shape.get("width", 0)))
                elif isinstance(frame_shape, (list, tuple)) and len(frame_shape) >= 2:
                    h, w = int(frame_shape[0]), int(frame_shape[1])
                else:
                    # if it's an ndarray shape
                    h, w = int(frame_shape[0]), int(frame_shape[1])
                # clip
                poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            except Exception:
                # ignore clipping if shape can't be interpreted
                pass

        return poly.astype(np.float32)

    def save_inside_region(self, inside_region, region_depth=50.0, region_diag=None):
        """
        Persist inside_region and diagnostics into self.curve_data and write to disk.
        Rounds numeric values to 2 decimal places as required.
        """
        if self.curve_data is None:
            self.curve_data = {}

        # store polygon as list of [x,y]
        self.curve_data["inside_region"] = inside_region.tolist()
        self.curve_data["region_depth"] = float(region_depth)
        self.curve_data["region_method"] = "normal_offset"
        if region_diag is None:
            region_diag = {"method": "normal_offset", "region_depth": float(region_depth), "num_points": int(len(inside_region)), "status": "ok"}
        self.curve_data["region_diagnostics"] = region_diag

        # Round all floats to 2 decimals before saving
        rounded = CurveManager._round_floats(self.curve_data, decimals=2)
        self.save_curve_config(rounded)
        # also keep in-memory as non-rounded floats for further computation (but apply rounding to stored fields)
        # ensure curve_data fields are set to rounded values for consistency
        self.curve_data = rounded

    # -------------------------
    # VISUALIZATION
    # -------------------------
    def visualize_region(self, frame, color=(0, 200, 0), alpha=0.2, draw_arrows=True, arrow_len=12):
        """
        Overlay the inside_region (if present in self.curve_data) onto the frame.
        - frame: BGR image (numpy array)
        - color: (B,G,R) tuple
        - alpha: fill transparency (0-1)
        - draw_arrows: small arrows pointing into the IN side (if orientation known)
        - arrow_len: pixel length of arrow glyphs
        Returns blended image (copy). Does not mutate original frame.
        """
        out = frame.copy()
        if not self.curve_data:
            return out

        inside = self.curve_data.get("inside_region", None)
        if inside is None:
            return out

        poly = np.array(inside, dtype=np.int32)
        if poly.ndim != 2 or poly.shape[0] < 3:
            return out

        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Draw polygon outline
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Optionally draw arrows along the curve showing IN direction
        if draw_arrows:
            orientation = self.curve_data.get("orientation", None)
            curve_pts = np.array(self.curve_data.get("curve_points", []), dtype=np.float32)
            if curve_pts.shape[0] >= 2:
                # compute normals again to determine arrow direction
                _, normals = CurveManager._compute_tangents_normals(curve_pts)
                # if orientation known choose correct normal sign; else draw symmetric arrows both ways
                if orientation is None:
                    # draw small arrows both sides (neutral)
                    for i, p in enumerate(curve_pts):
                        p_int = (int(p[0]), int(p[1]))
                        n = normals[i]
                        p_out1 = (int(p[0] + n[0] * arrow_len), int(p[1] + n[1] * arrow_len))
                        p_out2 = (int(p[0] - n[0] * arrow_len), int(p[1] - n[1] * arrow_len))
                        cv2.arrowedLine(out, p_int, p_out1, color, 1, tipLength=0.25)
                        cv2.arrowedLine(out, p_int, p_out2, color, 1, tipLength=0.25)
                else:
                    sign = 1 if int(orientation) == 1 else -1
                    for i, p in enumerate(curve_pts):
                        p_int = (int(p[0]), int(p[1]))
                        n = normals[i] * sign
                        p_out = (int(p[0] + n[0] * arrow_len), int(p[1] + n[1] * arrow_len))
                        cv2.arrowedLine(out, p_int, p_out, color, 1, tipLength=0.3)

        return out

    # -------------------------
    # Backwards-compatible simple build (kept for older callers)
    # -------------------------
    @staticmethod
    def legacy_build_inside_region(curve_points, in_direction, frame_shape):
        """Original axis-aligned extension fallback (kept for compatibility)."""
        h, w = frame_shape
        if in_direction in ["toward_cam", "away_from_cam"]:
            bottom, top = h, 0
            if in_direction == "toward_cam":
                extension = np.array([[curve_points[0, 0], bottom], [curve_points[-1, 0], bottom]])
            else:
                extension = np.array([[curve_points[0, 0], top], [curve_points[-1, 0], top]])
        elif in_direction in ["left", "right"]:
            left, right = 0, w
            if in_direction == "left":
                extension = np.array([[left, curve_points[0, 1]], [left, curve_points[-1, 1]]])
            else:
                extension = np.array([[right, curve_points[0, 1]], [right, curve_points[-1, 1]]])
        else:
            raise ValueError(f"Unknown IN_direction: {in_direction}")

        inside_region = np.concatenate((curve_points, extension), axis=0)
        return inside_region.astype(np.float32)
