import os
import glob
import numpy as np
import mediapipe as mp
from utils import compare_frames, options, check_frames, get_normilised_3d_points
from utils import compare_with_keras
import tqdm
import cv2
import random
from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

def get_confusion_matrix(frames_paths, model, threshold=0.02):

    
    n = len(frames_paths)
    dist_matrix = np.zeros((n, n))
    gesture_names = [os.path.basename(p).split('.')[0] for p in frames_paths]
    for i in tqdm(range(n)):
        for j in range(n):
            # You can try different metrics:
            # dist_matrix[i,j] = euclidean(norm_templates[i], norm_templates[j])
            label_1 = os.path.basename(frames_paths[i]).split('-')[0]
            label_2 = os.path.basename(frames_paths[j]).split('-')[0]

            if label_1 == label_2:
                label = 1 # same gestures
            else:
                label = 0    

            procruste, p2p_dist = compare_frames(frames_paths[i],
                                                           frames_paths[j],
                                                             landmarker)
            predict = model.predict(np.array(procruste).reshape((1,5)), verbose=0)[0]

            #if np.amax(procruste) > threshold:
            if predict < 0.5:
                predict_label = 0 # gestures are different
            else:
                predict_label = 1 # gestures are the same    

            dist_matrix[i,j] = bool(label==predict_label)
            #dist_matrix[i,j] = label
    correct = np.sum(dist_matrix)      
    print(correct/dist_matrix.size)
    
    # ─── Visualize ───
    plt.figure(figsize=(11, 9))
    sns.heatmap(dist_matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",           # or "viridis", "Blues", "magma"
                xticklabels=gesture_names,
                yticklabels=gesture_names,
                vmin=0, vmax=1)
    plt.title(f"Pairwise Gesture Similarity Matrix\n(weighted with th={threshold})")
    plt.xlabel("Frame name")
    plt.ylabel("Frame name")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_mtr.png')

    #plt.show()
    
if __name__ == "__main__":

    #frames_paths = glob.glob('C:\\Work\\gestures\\dataset\\Simple-gestures\\frames\\*jpg')
    #frames_paths = glob.glob('<PATH-TO-DATASET>\\Simple-gestures\\frames\\*jpg')

    HandLandmarker = mp.tasks.vision.HandLandmarker
    landmarker = HandLandmarker.create_from_options(options)
    model_embedder = load_model('artefacts/similarity_model-99.38.keras')
    '''
        
    for t in range(1, 20): # procruste
    #for t in range(8, 30): # procruste
    
        #print(t/200) # procruste
        th = t/200
        print(th, end=' ')
        get_confusion_matrix(frames_paths, th)
    '''    
    th = 0.04
    get_confusion_matrix(frames_paths, model_embedder, th)
    
    
    

 


