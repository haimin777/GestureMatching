from flask import Flask, render_template, request, jsonify
import cv2
import glob
import os
import numpy as np
import time
import mediapipe as mp
from utils import options, auto_rotate_frame, get_normilised_3d_points, p2p_distance
import mediapipe as mp
from scipy.spatial import procrustes


HandLandmarker = mp.tasks.vision.HandLandmarker


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"

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

   

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)  # frames per 0.5s

    count = 0
    saved = 0
    targets = []
    # clear folder
    [os.remove(p) for p in glob.glob('frames/*')]


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame = auto_rotate_frame(frame, exif_orientation=8)   # 8 = 90° left (most common for phones)
            out_path = os.path.join(FRAME_FOLDER, f"frame_{saved}.jpg")
            #vec, image, draw_data = get_rotation_invariant_vector(frame)
            normalized = get_normilised_3d_points(frame, landmarker)
            if normalized.shape == (21, 3):
                targets.append(normalized)
                saved += 1
                cv2.imwrite(out_path, frame)
                print('saved')

        count += 1

    cap.release()

    np.save('target.npy', targets)
    return f"Saved {saved} frames to folder '{FRAME_FOLDER}'"


# ----------------------------------------------------------
# 2️⃣ MediaPipe Hand Landmarks for the whole video
# ----------------------------------------------------------
@app.route("/hand_landmarks", methods=["POST"])
def hand_landmarks():
    video = request.files["video"]
    filename = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(filename)
    targets = np.load('target.npy')
    template_ind = 0
    template = targets[template_ind]
    [os.remove(p) for p in glob.glob('match/*')]
    cap = cv2.VideoCapture(filename)
    results_all = []

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % 2  == 0: 
                frame = auto_rotate_frame(frame, exif_orientation=8) 
                vec = get_normilised_3d_points(frame, landmarker)
                if vec.shape != (21,3):
                    print('empty frame', frame_id)
    
            _, _, disparity_1 = procrustes(vec, template)

            p2p, _ = p2p_distance(vec, template)
            p2p = round(p2p, 3)
            disparity_1 = round(disparity_1,3)
            if p2p < 0.25 and disparity_1 < 0.1:
                templ_img = cv2.imread(f'frames/frame_{template_ind}.jpg')
                cv2.putText(templ_img, f"TEMPLATE {template_ind}",
                             (110, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                res_img = np.concatenate([templ_img, frame], 1)
                res_img = cv2.resize(res_img, (512, 512)) 
                cv2.putText(res_img, f"P2P dist {p2p}",
                             (110, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(res_img, f"Procrust dist {disparity_1}",
                             (110, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imwrite(f'match/Match-{template_ind}.jpg', res_img)

                print(f'Match {template_ind}')
                results_all.append({'frame_id': frame_id,
                                    'p2p':p2p,
                                    'procruste_dist': disparity_1 })
                template_ind +=1
                if template_ind < len(targets):
                    template = targets[template_ind]
                else:
                    print('Full match')
                    break

            frame_id += 1

    cap.release()
    return jsonify(results_all)


if __name__ == "__main__":
    app.run(debug=True)
