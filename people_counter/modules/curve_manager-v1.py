import numpy as np
import json
import cv2

class CurveManagerV1:
    def __init__(self, config_path="config/curve_config.json"):
        self.config_path = config_path
        self.curve_points = []
        self.camera_orientation = None
        self.IN_direction = None
        self.frame = None
        self.done_drawing = False

    def create_curve(self, frame):
        """Interactive curve creation on the given frame."""
        self.frame = frame.copy()
        clone = frame.copy()
        self.curve_points = []
        self.done_drawing = False

        cv2.namedWindow("Define Curve")
        cv2.setMouseCallback("Define Curve", self._click_callback)

        instructions = "Click points to define curve. Enter=Finish, Esc=Cancel"
        while True:
            disp = self.frame.copy()
            cv2.putText(disp, instructions, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)
            # Draw points
            for pt in self.curve_points:
                cv2.circle(disp, tuple(pt), 5, (0,0,255), -1)
            # Draw lines
            if len(self.curve_points) > 1:
                cv2.polylines(disp, [np.array(self.curve_points)], False, (0,255,0), 2)

            cv2.imshow("Define Curve", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter: finish
                if len(self.curve_points) < 2:
                    print("⚠️ Need at least 2 points to define curve.")
                    continue
                break
            elif key == 27:  # Esc: cancel
                print("Curve creation cancelled.")
                self.curve_points = []
                cv2.destroyWindow("Define Curve")
                return None

        cv2.destroyWindow("Define Curve")
        # Ask camera orientation
        self._select_camera_orientation(frame)
        # Build IN_direction
        self._map_IN_direction()
        config = {
            "curve_points": [list(pt) for pt in self.curve_points],
            "camera_orientation": self.camera_orientation,
            "IN_direction": self.IN_direction
        }
        return config

    def _click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.curve_points.append([x, y])

    def _select_camera_orientation(self, frame):
        prompt_window = "Select Camera Orientation"
        options = {
            ord('0'): "front-facing",
            ord('1'): "back-facing",
            ord('2'): "left-side",
            ord('3'): "right-side",
            ord('4'): "overhead"
        }
        instructions = [
            "Select camera orientation:",
            "0: Front-facing",
            "1: Back-facing",
            "2: Left-side",
            "3: Right-side",
            "4: Overhead"
        ]
        while True:
            disp = frame.copy()
            y0 = 30
            for line in instructions:
                cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,0), 2)
                y0 += 30
            cv2.imshow(prompt_window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in options:
                self.camera_orientation = options[key]
                print(f"Camera orientation selected: {self.camera_orientation}")
                break
        cv2.destroyWindow(prompt_window)

    def _map_IN_direction(self):
        mapping = {
            "front-facing": "toward_cam",
            "back-facing": "away_from_cam",
            "left-side": "left",
            "right-side": "right",
            "overhead": "left"  # user can adjust manually if needed
        }
        self.IN_direction = mapping.get(self.camera_orientation, "toward_cam")
        print(f"IN_direction set to: {self.IN_direction}")

    def save_curve_config(self, config):
        """Save the created curve to JSON."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Curve config saved to {self.config_path}")

    def load_curve_config(self):
        """Load curve config if exists."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            self.curve_points = np.array(data["curve_points"], dtype=np.float32)
            self.camera_orientation = data.get("camera_orientation", None)
            self.IN_direction = data.get("IN_direction", None)
            return data
        except FileNotFoundError:
            print(f"⚠️ Config file {self.config_path} not found.")
            return None
