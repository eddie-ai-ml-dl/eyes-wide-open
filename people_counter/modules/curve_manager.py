import os
import json
import cv2
import numpy as np
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
        """
        if not self.curve_data:
            raise ValueError("Curve must be loaded or created before determining orientation.")

        curve_np=np.array(self.curve_data["curve_points"], dtype=np.float32)
        in_dir=self.curve_data.get("IN_direction", "auto")

        diag=None
        if in_dir is None or in_dir=="auto":
            # Auto-detect orientation
            orientation, diag=auto_orient_curve(curve_np, sample_anchors, eps=3.0, min_crossings=1)
            print(f"🧭 Auto-detected curve orientation: {orientation}")
            print("Diagnostics:", diag)

            # Save orientation and diagnostics
            self.curve_data["orientation"]=int(orientation)
            self.curve_data["camera_orientation"]=diag.get("camera_convention", None)
            self.curve_data["orientation_diagnostics"]=diag

            # --- Resolve IN_direction based on orientation ---
            resolved_in_dir="toward_cam" if orientation==1 else "away_from_cam"
            self.curve_data["IN_direction"]=resolved_in_dir

            # Save back to config
            self.save_curve_config(self.curve_data)
        else:
            # Already set in config
            orientation=1 if in_dir in ["toward_cam", "left", "right"] else -1
            resolved_in_dir=in_dir
            print(f"➡️ Using orientation from config IN_direction='{in_dir}': {orientation}")

        return orientation, diag, resolved_in_dir

    # -------------------------
    # REGION GEOMETRY
    # -------------------------
    @staticmethod
    def build_inside_region(curve_points, in_direction, frame_shape):
        """Construct INSIDE polygon region based on curve orientation."""
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
