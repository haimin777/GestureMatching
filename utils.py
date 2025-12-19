
import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import glob
from scipy.spatial import procrustes


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FINGERTIP_IDS = [4, 8, 12, 16, 20]
FRAME_FOLDER = "frames"



options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="artefacts/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

def auto_rotate_frame(frame, exif_orientation=None):
    if exif_orientation == 8:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 3:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif exif_orientation == 6:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def get_normilised_3d_points(image, landmarker):
    if type(image) == str:
        
        img = cv2.imread(image)
    else:
        img = image
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = landmarker.detect(mp.Image(data=rgb, image_format=mp.ImageFormat.SRGB))
    if not result.hand_landmarks:
        return np.array([0])

    lm = result.hand_landmarks[0]                     # first hand only
    
    pts = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(21)])
    #pts = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in [0, 1, 4, 5, 8, 12, 16, 20]])


    # 1. Translate wrist (0) → origin
    wrist = pts[0]
    pts = pts - wrist

    # 2. Align middle-finger direction (wrist → middle tip = landmark 12) to +Y axis
    middle_dir = pts[12]  # wrist → middle fingertip
    middle_dir /= np.linalg.norm(middle_dir) + 1e-8

    # Build rotation matrix: middle finger → [0,1,0]
    y_axis = np.array([0., 1., 0.])
    v = np.cross(middle_dir, y_axis)
    c = np.dot(middle_dir, y_axis)
    if abs(c - 1) < 1e-6:  # already aligned
        R = np.eye(3)
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)  # Rodrigues formula

    # 3. Apply rotation + scale by distance 0→5
    pts_rot = pts @ R.T
    scale = np.linalg.norm(pts[5]) + 1e-8
    pts_norm = pts_rot / scale

    return pts_norm

def p2p_distance(gesture1, gesture2):
    """
    Computes average point-to-point Euclidean distance between 2 gestures.
    gesture1, gesture2: (21,3)
    """
    g1 = np.array(gesture1)
    g2 = np.array(gesture2)

    # Euclidean distance per keypoint
    dists = np.linalg.norm(g1 - g2, axis=1)
    weights = np.ones(21)
    weights[FINGERTIP_IDS] = 2.0

    #return np.mean(dists), dists  # (mean distance, per-point distances)    
    return np.average(dists, weights=weights), dists

def p2p_distance_fingers(gesture1, gesture2):
    """
    Computes average point-to-point Euclidean distance between 2 gestures.
    gesture1, gesture2: (21,3)
    """
    g1 = np.array(gesture1)
    g2 = np.array(gesture2)

    # Euclidean distance per keypoint
    dists = np.linalg.norm(g1 - g2, axis=1)
    fing_1 = [dists[i] for i in [0, 1, 2, 3, 4]]

    return np.mean(fing_1), fing_1  # (mean distance, per-point distances)    

def register_gesture(video_path:str, landmarker):
    '''
    extract frames and landmarks
    '''
    cap = cv2.VideoCapture(video_path)
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
    return saved

def match_gestures(filename, landmarker, targets):
    template_ind = 0
    template = targets[template_ind]
    cap = cv2.VideoCapture(filename)
    results_all = []
    results_debug = []

    '''
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
    '''    
    if True:

        frame_id = 0
        finger_1_points = [0, 1, 2, 3, 4]
        finger_2_points = [0, 5, 6, 7, 8]
        finger_3_points = [0, 9, 10, 11, 12]
        finger_4_points = [0, 13, 14, 15, 16]
        finger_5_points = [0, 17, 18, 19, 20]



        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % 2  == 0: 
            #if True:
                frame = auto_rotate_frame(frame, exif_orientation=8) 
                #frame = cv2.resize(frame, (512, 512))
                vec = get_normilised_3d_points(frame, landmarker)
                if vec.shape != (21,3):
                    print('empty frame', frame_id)
                    continue
    
                #_, _, disparity_1 = procrustes(vec, template)
                fing_1_vec = [vec[i] for i in finger_1_points]
                fing_1_template = [template[i] for i in finger_1_points]

                fing_2_vec = [vec[i] for i in finger_2_points]
                fing_2_template = [template[i] for i in finger_2_points]

                fing_3_vec = [vec[i] for i in finger_3_points]
                fing_3_template = [template[i] for i in finger_3_points]

                fing_4_vec = [vec[i] for i in finger_4_points]
                fing_4_template = [template[i] for i in finger_4_points]

                fing_5_vec = [vec[i] for i in finger_5_points]
                fing_5_template = [template[i] for i in finger_5_points]

                _, _, disparity_1 = procrustes(fing_1_vec, fing_1_template)
                _, _, disparity_2 = procrustes(fing_2_vec, fing_2_template)
                _, _, disparity_3 = procrustes(fing_3_vec, fing_3_template)
                _, _, disparity_4 = procrustes(fing_4_vec, fing_4_template)
                _, _, disparity_5 = procrustes(fing_5_vec, fing_5_template)

                p2p, dist = p2p_distance(vec, template)
                #p2p, dist = p2p_distance_fingers(vec, template)
                max_p2p_dist = np.amax(dist)
                p2p = round(p2p, 4)
                disparity_1 = round(disparity_1, 4)
                disparity_2 = round(disparity_2, 4)
                results_debug.append({'max_p2p_dist': max_p2p_dist,
                                    'p2p': p2p,
                                    'pr_1': disparity_1,
                                    'pr_2': disparity_2,
                                    'pr_3': disparity_3})
                print(max_p2p_dist, p2p, disparity_1, disparity_2)
                if max(disparity_1,
                        disparity_2,
                        disparity_3,
                        disparity_4,
                        disparity_5) < 0.02:
                    templ_img = cv2.imread(f'frames/frame_{template_ind}.jpg')
                    #templ_img = cv2.resize(templ_img, (512, 512))
                    cv2.putText(templ_img, f"TEMPLATE {template_ind}",
                                (110, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    res_img = np.concatenate([templ_img, frame], 1)
                    res_img = cv2.resize(res_img, (512, 512)) 
                    cv2.putText(res_img, f"P2P max dist {max_p2p_dist}",
                                (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(res_img, f"Procrust dist fing1 {disparity_1}",
                                (70, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(res_img, f"Procrust dist fing2 {disparity_2}",
                                (70, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(res_img, f"Procrust dist fing3 {disparity_3}",
                                (70, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(res_img, f"p2p aver {p2p}",
                                (70, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    targ_len = len(targets)
    res_len = len(results_all)
    debug_logs = pd.DataFrame(results_debug)
    debug_logs.to_csv('debug_logs.csv', index=None)
    return targ_len, res_len
    