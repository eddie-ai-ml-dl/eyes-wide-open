import cv2
from modules.curve_manager import CurveManager

cap = cv2.VideoCapture("../data/videos/People Entering And Exiting Mall Stock Footage.mp4")
ret, frame = cap.read()
cap.release()

cm = CurveManager("config/curve_config_02.json")
config = cm.create_curve(frame)
if config:
    cm.save_curve_config(config)
    print("Curve created and saved:", config)
