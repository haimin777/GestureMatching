from flask import Flask, render_template, request, jsonify
import cv2
import glob
import os
import numpy as np
import time
import mediapipe as mp
from utils import options, auto_rotate_frame, get_normilised_3d_points, p2p_distance, p2p_distance_fingers
from utils import register_gesture, match_gestures
import mediapipe as mp
from scipy.spatial import procrustes


HandLandmarker = mp.tasks.vision.HandLandmarker


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
FINGERTIP_IDS = [4, 8, 12, 16, 20]


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
landmarker = HandLandmarker.create_from_options(options)


@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------
# 1️⃣ Extract frames every 0.5 seconds
# ----------------------------------------------------------
@app.route("/extract_frames", methods=["POST"])
def extract_frames():
    video = request.files["video"]
    filename = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(filename)

    registered_gestures_num = register_gesture(filename, landmarker)
    return f"Saved {registered_gestures_num} frames to folder '{FRAME_FOLDER}'"


# ----------------------------------------------------------
# 2️⃣ MediaPipe Hand Landmarks for the whole video
# ----------------------------------------------------------
@app.route("/hand_landmarks", methods=["POST"])
def hand_landmarks():
    video = request.files["video"]
    filename = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(filename)
    targets = np.load('target.npy')
    #template_ind = 0
    #template = targets[template_ind]
    [os.remove(p) for p in glob.glob('match/*')]
    
    targ_len, res_len = match_gestures(filename, landmarker, targets)

    if targ_len == res_len:
        return f"""s<h2>Full match. \
        Registerd gestures num {targ_len}, detected num {res_len} \n Matched frames in folder MATCH </h2> \
        <form action="/" method="get"> \
            <button type="submit">Home</button>
        </form>
        """
    else:

        #return jsonify(results_all)
        return f"""<h2>NOT matched. Registerd gestures num {targ_len}, detected num {res_len}</h2> 
        <form action="/" method="get"> \
            <button type="submit">Home</button>
        </form>
        """
        

@app.route("/record_video", methods=["POST"])
def record_video():
    output_path = os.path.join(UPLOAD_FOLDER, "recorded_5s.mp4")

    cap = cv2.VideoCapture(0)  # 0 = laptop camera

    if not cap.isOpened():
        return "Cannot open camera"

    fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()

    while time.time() - start_time < 5:  # record 5 seconds
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

    cap.release()
    out.release()
    # register gestures from saved video
    [os.remove(p) for p in glob.glob('match/*')]

    registered_gestures_num = register_gesture(output_path, landmarker)
    

    return f"""
    <h2>Recording finished!</h2>
    <p>Saved video: recorded_5s.mp4</p>
    <p>Saved gestures: {registered_gestures_num}</p>

    <form action="/" method="get">
        <button type="submit">Home</button>
    </form>
    """

@app.route("/match_video", methods=["POST"])
def match_video():
    output_path = os.path.join(UPLOAD_FOLDER, "recorded_5s_for_match.mp4")

    cap = cv2.VideoCapture(0)  # 0 = laptop camera

    if not cap.isOpened():
        return "Cannot open camera"

    fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()

    while time.time() - start_time < 5:  # record 5 seconds
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

    cap.release()
    out.release()
    # Match gestures from saved video
    targets = np.load('target.npy')
    [os.remove(p) for p in glob.glob('match/*')]
    targ_len, res_len = match_gestures(output_path, landmarker, targets)

    if targ_len == res_len:
        return f"""s<h2>Full match. \
        Registerd gestures num {targ_len}, detected num {res_len} \n Matched frames in folder MATCH </h2> \
        <form action="/" method="get"> \
            <button type="submit">Home</button>
        </form>
        """
    else:

        #return jsonify(results_all)
        return f"""<h2>NOT matched. Registerd gestures num {targ_len}, detected num {res_len}</h2> 
        <form action="/" method="get"> \
            <button type="submit">Home</button>
        </form>
        """


if __name__ == "__main__":
    app.run(debug=True)
