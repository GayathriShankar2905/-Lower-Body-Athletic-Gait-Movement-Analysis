from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
video_path = r"C:\Users\GAYATHRI\OneDrive\Videos\Captures\Rio Replay_ Men's +105kg Weightlifting Final - YouTube - Google Chrome 2025-11-01 14-38-22.mp4"
output_folder = "output_metrics"
model = YOLO("yolov8n-pose.pt")
os.makedirs(output_folder, exist_ok=True)
# ----------------------------------------

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
angle_data, bar_positions, left_ankles, right_ankles = [], [], [], []

frame_idx = 0
print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        kps = results[0].keypoints.xy[0].cpu().numpy()

        # COCO indices
        L_SH, L_EL, L_WR = 5, 7, 9
        R_SH, R_EL, R_WR = 6, 8, 10
        L_HP, L_KN, L_AN = 11, 13, 15
        R_HP, R_KN, R_AN = 12, 14, 16

        # Angles
        left_elbow = calculate_angle(kps[L_SH], kps[L_EL], kps[L_WR])
        right_elbow = calculate_angle(kps[R_SH], kps[R_EL], kps[R_WR])
        left_knee = calculate_angle(kps[L_HP], kps[L_KN], kps[L_AN])
        right_knee = calculate_angle(kps[R_HP], kps[R_KN], kps[R_AN])

        # Barbell midpoint
        bar_x = int((kps[L_WR][0] + kps[R_WR][0]) / 2)
        bar_y = int((kps[L_WR][1] + kps[R_WR][1]) / 2)
        bar_positions.append((frame_idx, bar_x, bar_y))

        # Track ankles
        left_ankles.append(kps[L_AN])
        right_ankles.append(kps[R_AN])

        # Annotate
        cv2.circle(annotated, (bar_x, bar_y), 6, (0,255,255), -1)
        cv2.putText(annotated, f"L-Knee: {left_knee:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated, f"L-Elbow: {left_elbow:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if frame_idx % 50 == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_idx}.jpg"), annotated)

        angle_data.append({
            "frame": frame_idx,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "bar_x": bar_x,
            "bar_y": bar_y
        })

    cv2.imshow("Biomechanics Analysis", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ------------------ METRICS ------------------
if not angle_data:
    print("No data extracted.")
    exit()

df = pd.DataFrame(angle_data)
df["time_sec"] = df["frame"] / fps
df["bar_velocity"] = df["bar_y"].diff().fillna(0) * -1

# ---- Stride, Cadence, Gait Speed (based on ankle motion) ----
left_y = [a[1] for a in left_ankles if len(a)==2]
right_y = [a[1] for a in right_ankles if len(a)==2]

# Convert pixel distance to "stride" (approx.)
if len(left_y) > 5:
    stride_pix = np.max(left_y) - np.min(left_y)
    stride_length = stride_pix / 100  # scale (approx meters per pixel)
else:
    stride_length = 0.0

# Simple step detection using ankle movement pattern
peaks = np.sum(np.diff(np.sign(np.diff(left_y))) < 0)
cadence = (peaks / (len(df)/fps)) * 60 if len(df)>5 else 0
gait_speed = stride_length * (cadence / 120)  # (2 steps per stride)

# ---- Add summary metrics ----
summary = {
    "Average_Left_Knee_Angle": np.mean(df["left_knee"]),
    "Average_Right_Knee_Angle": np.mean(df["right_knee"]),
    "Average_Bar_Velocity": np.mean(df["bar_velocity"]),
    "Estimated_Stride_Length_m": stride_length,
    "Estimated_Cadence_spm": cadence,
    "Estimated_Gait_Speed_mps": gait_speed
}

df.to_csv(os.path.join(output_folder, "full_metrics.csv"), index=False)
pd.DataFrame([summary]).to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)

print("\n✅ CSV files created:")
print(f"  - {output_folder}/full_metrics.csv  (frame-by-frame data)")
print(f"  - {output_folder}/summary_metrics.csv  (quantitative summary)\n")

# ------------------ PLOTS ------------------
plt.figure(figsize=(10,5))
plt.plot(df["time_sec"], df["left_knee"], label="Left Knee Angle")
plt.plot(df["time_sec"], df["bar_y"]/5, label="Bar Height (scaled)")
plt.xlabel("Time (s)")
plt.ylabel("Angle / Height")
plt.title("Knee Motion & Bar Trajectory")
plt.legend(); plt.grid()
plt.savefig(os.path.join(output_folder, "motion_graph.png"))
plt.show()
