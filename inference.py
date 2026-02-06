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

def main(frame_path_1, frame_path_2, landmarker, comparison_model, th=0.04):
    
    procruste, p2p_dist = compare_frames(frame_path_1,
                                         frame_path_2,
                                        landmarker)
    # compare without model
    if np.amax(procruste) > th:
        predict_label_dist = 0 # gestures are different
    else:
        predict_label_dist = 1 # gestures are the same 
        
    #compare with model over procruste fingerwise distance  
    predict = comparison_model.predict(np.array(procruste).reshape((1,5)), verbose=0)[0] 
                
    if predict < 0.5:
        predict_label_model = 0 # gestures are different
    else:
        predict_label_model = 1 # gestures are the same 
        
    print('Labels: 1: Same, 0: Different. \n Comparison results on landmark distances: ', predict_label_dist)
    print(f'Comparison results on landmark distances + model: {predict_label_model}, score {predict}' )   
    
    
if __name__ == "__main__":
    HandLandmarker = mp.tasks.vision.HandLandmarker
    landmarker = HandLandmarker.create_from_options(options)
    comparison_model = load_model('artefacts/similarity_model-99.38.keras')
    
    frame_1 = 'frames/4-5.jpg'
    frame_2 = 'frames/5-1.jpg'
    frame_3 = 'frames/4-2.jpg'
    
    main(frame_1, frame_2, landmarker, comparison_model)
    main(frame_1, frame_3, landmarker, comparison_model)

    
    
    
         
                
    