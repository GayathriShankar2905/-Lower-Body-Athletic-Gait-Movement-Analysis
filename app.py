import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os

st.set_page_config(page_title="🏋️‍♂️ Weightlifting Biomechanics", layout="wide")
st.title("🏋️‍♂️ Lower-Body Gait & Movement Analysis using Open-Source Tools")
st.markdown("Upload a **weightlifting video** to extract joint angles, barbell trajectory, and gait metrics.")

# ------------------- Helper Functions -------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ------------------- File Upload -------------------
video_file = st.file_uploader("🎥 Upload Video", type=["mp4", "avi", "mov"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_file)

    run_btn = st.button("▶️ Run Analysis")
    if run_btn:
        st.info("Running YOLOv8 Pose Estimation... This may take a few minutes.")
        progress = st.progress(0)

        output_folder = "output_metrics"
        os.makedirs(output_folder, exist_ok=True)

        model = YOLO("yolov8n-pose.pt")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        angle_data, left_ankles, right_ankles = [], [], []

        frame_idx = 0
        sample_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            if results[0].keypoints is not None:
                kps = results[0].keypoints.xy[0].cpu().numpy()
                L_SH, L_EL, L_WR = 5, 7, 9
                R_SH, R_EL, R_WR = 6, 8, 10
                L_HP, L_KN, L_AN = 11, 13, 15
                R_HP, R_KN, R_AN = 12, 14, 16

                left_elbow = calculate_angle(kps[L_SH], kps[L_EL], kps[L_WR])
                right_elbow = calculate_angle(kps[R_SH], kps[R_EL], kps[R_WR])
                left_knee = calculate_angle(kps[L_HP], kps[L_KN], kps[L_AN])
                right_knee = calculate_angle(kps[R_HP], kps[R_KN], kps[R_AN])

                bar_x = int((kps[L_WR][0] + kps[R_WR][0]) / 2)
                bar_y = int((kps[L_WR][1] + kps[R_WR][1]) / 2)

                left_ankles.append(kps[L_AN])
                right_ankles.append(kps[R_AN])

                angle_data.append({
                    "frame": frame_idx,
                    "left_elbow": left_elbow,
                    "right_elbow": right_elbow,
                    "left_knee": left_knee,
                    "right_knee": right_knee,
                    "bar_x": bar_x,
                    "bar_y": bar_y
                })

                if frame_idx % 50 == 0:
                    annotated = results[0].plot()
                    sample_frames.append(annotated)

            frame_idx += 1
            if frame_idx % 10 == 0:
                progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        progress.progress(1.0)
        st.success("✅ Analysis complete!")

        if not angle_data:
            st.warning("No pose data detected. Try a clearer video.")
        else:
            df = pd.DataFrame(angle_data)
            df["time_sec"] = df["frame"] / fps
            df["bar_velocity"] = df["bar_y"].diff().fillna(0) * -1

            left_y = [a[1] for a in left_ankles if len(a)==2]
            right_y = [a[1] for a in right_ankles if len(a)==2]

            if len(left_y) > 5:
                stride_pix = np.max(left_y) - np.min(left_y)
                stride_length = stride_pix / 100
            else:
                stride_length = 0.0

            peaks = np.sum(np.diff(np.sign(np.diff(left_y))) < 0)
            cadence = (peaks / (len(df)/fps)) * 60 if len(df)>5 else 0
            gait_speed = stride_length * (cadence / 120)

            summary = {
                "Average Left Knee Angle": round(np.mean(df["left_knee"]), 2),
                "Average Right Knee Angle": round(np.mean(df["right_knee"]), 2),
                "Average Bar Velocity": round(np.mean(df["bar_velocity"]), 2),
                "Estimated Stride Length (m)": round(stride_length, 3),
                "Estimated Cadence (steps/min)": round(cadence, 2),
                "Estimated Gait Speed (m/s)": round(gait_speed, 3)
            }

            df.to_csv(os.path.join(output_folder, "full_metrics.csv"), index=False)
            pd.DataFrame([summary]).to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)

            st.subheader("📊 Summary Metrics")
            st.dataframe(pd.DataFrame([summary]))

            # ----------- PLOT -----------
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["time_sec"], df["left_knee"], label="Left Knee Angle")
            ax.plot(df["time_sec"], df["bar_y"]/5, label="Bar Height (scaled)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle / Height")
            ax.set_title("Knee Motion & Bar Trajectory")
            ax.legend(); ax.grid()
            st.pyplot(fig)

            # ----------- Sample Annotated Frames -----------
            if sample_frames:
                st.subheader("📸 Annotated Frames")
                cols = st.columns(3)
                for i, frame_img in enumerate(sample_frames[:6]):
                    rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    cols[i % 3].image(rgb, caption=f"Frame {i*50}")

            st.success("Results saved to /output_metrics folder ✅")
