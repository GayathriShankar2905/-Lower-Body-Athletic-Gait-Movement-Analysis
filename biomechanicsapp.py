import streamlit as st
import cv2, os, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --------------------------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------------------------
st.set_page_config(page_title="🏋️ Weightlifting Biomechanics Lab", layout="wide")

# Sidebar navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Page 1 – Project Overview",
    "📘 Page 2 – Background & Theory",
    "🧠 Page 3 – Experiment",
    "📊 Page 4 – Results Dashboard"
])

# --------------------------------------------------------------------
# PAGE 1 – FRONT PAGE / INFO
# --------------------------------------------------------------------
if page.startswith("🏠"):
    st.title("🏋️‍♂️ Lower-Body Athletic Gait & Movement Analysis")
    st.markdown("""
    ### **SRM Institute of Science and Technology**
    **College of Engineering and Technology**  
    **School of Bioengineering**  
    **Department of Biomedical Engineering**  
    SRM Nagar, Kattankulathur – 603203, Chengalpattu District, Tamil Nadu  

    **Academic Year:** 2025-26 (ODD: FT)  
    **Course Code & Title:** 21BMC401J – *BIOMECHANICS*  
    **Year & Semester:** IV Year, VII Semester  

    ---
    **Done By:**  
    👩‍🎓 **Gayathri S.H**  
    B.Tech Biomedical Engineering  
    📧 *pcmjs.gayathri@gmail.com*  

    ---
    ### **Brief Description**
    This project focuses on analyzing human motion using open-source computer-vision tools.
    A YOLO-based pose-estimation model extracts kinematic parameters such as
    stride length, cadence, gait speed, joint angles, and barbell trajectory from video data
    of athletic movements (e.g., weightlifting).  
    The goal is to provide quantitative insights for performance enhancement and technique improvement.
    """)

    st.markdown("### ➡️ Use the sidebar to explore: Background, Experiment, and Results pages.")

# --------------------------------------------------------------------
# PAGE 2 – BACKGROUND / THEORY
# --------------------------------------------------------------------
elif page.startswith("📘"):
    st.title("📘 Background & Theory")

    st.header("🦵 What is Gait?")
    st.write("""
    Gait refers to the pattern of limb movements during locomotion.  
    One gait cycle begins with heel-strike of one foot and ends with its next heel-strike.
    It consists of:
    * **Stance Phase (≈60%)** – foot in contact with ground.  
    * **Swing Phase (≈40%)** – foot moves forward for next contact.
    """)

    st.header("⚙️ Why Analyze Gait and Athletic Motion?")
    st.write("""
    * **Clinical Rehabilitation** – Assess balance, posture, and mobility.  
    * **Sports Biomechanics** – Optimize performance and prevent injury.  
    * **Assistive Device Design** – Develop prosthetics mimicking natural motion.
    """)

    st.header("🏋️ Weightlifting Biomechanics")
    st.write("""
    In Olympic lifts (Snatch, Clean & Jerk), the athlete’s hip, knee, and shoulder joints
    coordinate to move the barbell efficiently.
    Tracking joint angles and bar trajectory reveals force generation and symmetry,
    aiding performance improvement.
    """)

    st.header("📏 Quantitative Analysis")
    st.table(pd.DataFrame({
        "Parameter": [
            "Stride Length", "Cadence", "Gait Speed",
            "Joint Angles", "Barbell Trajectory", "Lifting Velocity"
        ],
        "Description": [
            "Distance between successive heel strikes",
            "Steps per minute",
            "Stride Length × Cadence / 120",
            "Angle between limb segments",
            "Path of barbell centroid across frames",
            "Rate of vertical barbell displacement (Δy/Δt)"
        ]
    }))

    st.header("🧰 Libraries Used")
    st.markdown("""
    * **OpenCV** – Video processing  
    * **Ultralytics YOLOv8-Pose** – Pose estimation  
    * **NumPy / Pandas** – Computation & data management  
    * **Matplotlib** – Visualization  
    * **Streamlit** – Interactive UI
    """)

# --------------------------------------------------------------------
# PAGE 3 – EXPERIMENT / INTERACTIVE LAB
# --------------------------------------------------------------------
elif page.startswith("🧠"):

    st.title("🧠 Interactive Experiment – Weightlifting & Gait Analysis")
    st.markdown("Upload a **weightlifting video** to extract joint angles, barbell trajectory, and gait metrics.")

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    video_file = st.file_uploader("🎥 Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        st.video(video_file)

        if st.button("▶️ Run Analysis"):
            st.info("Running YOLOv8 Pose Estimation … please wait.")
            progress = st.progress(0)

            model = YOLO("yolov8n-pose.pt")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx, angle_data, left_ankles, right_ankles, sample_frames = 0, [], [], [], []

            while True:
                ret, frame = cap.read()
                if not ret: break
                results = model(frame, verbose=False)
                if results[0].keypoints is not None:
                    kps = results[0].keypoints.xy[0].cpu().numpy()
                    L_SH,L_EL,L_WR,R_SH,R_EL,R_WR,L_HP,L_KN,L_AN,R_HP,R_KN,R_AN = 5,7,9,6,8,10,11,13,15,12,14,16
                    left_knee = calculate_angle(kps[L_HP], kps[L_KN], kps[L_AN])
                    right_knee = calculate_angle(kps[R_HP], kps[R_KN], kps[R_AN])
                    bar_x = int((kps[L_WR][0]+kps[R_WR][0])/2)
                    bar_y = int((kps[L_WR][1]+kps[R_WR][1])/2)
                    left_ankles.append(kps[L_AN]); right_ankles.append(kps[R_AN])
                    angle_data.append({
                        "frame": frame_idx,
                        "left_knee": left_knee,
                        "right_knee": right_knee,
                        "bar_x": bar_x, "bar_y": bar_y
                    })
                    if frame_idx % 50 == 0:
                        sample_frames.append(results[0].plot())
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress.progress(min(frame_idx/total_frames,1.0))
            cap.release()
            progress.progress(1.0)

            if not angle_data:
                st.warning("No pose data detected. Try a clearer video.")
            else:
                df = pd.DataFrame(angle_data)
                df["time_sec"] = df["frame"]/fps
                df["bar_velocity"] = df["bar_y"].diff().fillna(0)*-1

                left_y = [a[1] for a in left_ankles if len(a)==2]
                stride_pix = np.max(left_y)-np.min(left_y) if len(left_y)>5 else 0
                stride_length = stride_pix/100
                peaks = np.sum(np.diff(np.sign(np.diff(left_y)))<0)
                cadence = (peaks/(len(df)/fps))*60 if len(df)>5 else 0
                gait_speed = stride_length*(cadence/120)

                summary = {
                    "Avg Left Knee Angle (°)": round(np.mean(df["left_knee"]),2),
                    "Avg Right Knee Angle (°)": round(np.mean(df["right_knee"]),2),
                    "Avg Bar Velocity": round(np.mean(df["bar_velocity"]),2),
                    "Stride Length (m)": round(stride_length,3),
                    "Cadence (steps/min)": round(cadence,2),
                    "Gait Speed (m/s)": round(gait_speed,3)
                }

                st.session_state["results_df"] = df
                st.session_state["summary"] = summary
                st.success("✅ Analysis Complete! Proceed to Results Dashboard (page 4).")

# --------------------------------------------------------------------
# PAGE 4 – RESULTS DASHBOARD
# --------------------------------------------------------------------
elif page.startswith("📊"):

    st.title("📊 Results Dashboard – Analysis Summary")

    if "summary" not in st.session_state:
        st.warning("⚠️ Please run the experiment on Page 3 first.")
    else:
        summary = st.session_state["summary"]
        df = st.session_state["results_df"]

        st.subheader("📋 Summary Metrics")
        st.dataframe(pd.DataFrame([summary]))

        st.subheader("📈 Joint Angles & Bar Trajectory")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df["time_sec"], df["left_knee"], label="Left Knee Angle (°)")
        ax.plot(df["time_sec"], df["right_knee"], label="Right Knee Angle (°)")
        ax.plot(df["time_sec"], df["bar_y"]/5, label="Bar Height (Scaled)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle / Height"); ax.grid()
        ax.legend(); st.pyplot(fig)

        st.subheader("📂 Export Data")
        csv_full = df.to_csv(index=False).encode("utf-8")
        csv_summary = pd.DataFrame([summary]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Metrics CSV", data=csv_full, file_name="full_metrics.csv")
        st.download_button("⬇️ Download Summary Metrics CSV", data=csv_summary, file_name="summary_metrics.csv")

        st.markdown("""
        ---
        ### **Performance Insights**
        * High cadence + short stride → fast but less efficient motion  
        * Asymmetric angles → possible imbalance or technique error  
        * Sudden velocity spikes → explosive strength in lift phases
        ---
        """)

# --------------------------------------------------------------------
# END OF APP
# --------------------------------------------------------------------
