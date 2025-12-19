
import numpy as np
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FINGERTIP_IDS = [4, 8, 12, 16, 20]


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
    