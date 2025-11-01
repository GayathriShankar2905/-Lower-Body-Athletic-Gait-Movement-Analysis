from ultralytics import YOLO
import cv2
import numpy as np
import math
import pandas as pd

# ------------------ Configuration ------------------
video_path = r"C:\Users\GAYATHRI\OneDrive\Videos\Captures\Rio Replay_ Men's +105kg Weightlifting Final - YouTube - Google Chrome 2025-11-01 13-58-23.mp4"
model = YOLO("yolov8n-pose.pt")   # Lightweight YOLOv8 Pose model
output_csv = "pose_angles.csv"
# ---------------------------------------------------

# ---- Helper function: compute angle between 3 points ----
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

# ---- Video Processing ----
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Error: Could not open video file.")

frame_idx = 0
angle_data = []

print("Processing video... Press ESC to quit preview.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None:
        kps = results[0].keypoints.xy[0].cpu().numpy()  # keypoints for first detected person

        # Keypoint indices (COCO format)
        LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9
        RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
        LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15
        RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16

        # Calculate major joint angles
        left_elbow_angle = calculate_angle(kps[LEFT_SHOULDER], kps[LEFT_ELBOW], kps[LEFT_WRIST])
        right_elbow_angle = calculate_angle(kps[RIGHT_SHOULDER], kps[RIGHT_ELBOW], kps[RIGHT_WRIST])
        left_knee_angle = calculate_angle(kps[LEFT_HIP], kps[LEFT_KNEE], kps[LEFT_ANKLE])
        right_knee_angle = calculate_angle(kps[RIGHT_HIP], kps[RIGHT_KNEE], kps[RIGHT_ANKLE])

        # Store data
        angle_data.append({
            "frame": frame_idx,
            "left_elbow": left_elbow_angle,
            "right_elbow": right_elbow_angle,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle
        })

        # Display angles on frame
        cv2.putText(annotated_frame, f"L-Elbow: {left_elbow_angle}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated_frame, f"R-Elbow: {right_elbow_angle}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated_frame, f"L-Knee: {left_knee_angle}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated_frame, f"R-Knee: {right_knee_angle}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Weightlifting Pose Analysis", annotated_frame)

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

# ---- Save to CSV ----
if angle_data:
    df = pd.DataFrame(angle_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Joint angle data saved to {output_csv}")
else:
    print("\n⚠️ No pose data detected.")
