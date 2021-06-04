import numpy as np
import cv2

def segment_hand(frame, delta=10):
    """"
    This function extract the area of an image which contains a hand.
    @ Parameters:
        frame: the frame (image) to extract the hand region from
        delta: relaxation parameter (0:20) default value = 10
    @ Returns:
        roi: The region of interest which includes the hand
    """
    # Skin color limits
    lower_limit = np.array([0,40,40], dtype=np.uint8)
    upper_limit = np.array([20,255,255], dtype=np.uint8)
    # Convert color to HSV domain
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Decide the skin region
    skin_region = cv2.inRange(img, lower_limit, upper_limit)
    # Apply thresholding segmentation
    ret,thresholded = cv2.threshold(skin_region,0,255,cv2.THRESH_BINARY)
    # Find contours around the thresholded regions
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    # Apply convex hull algorithm
    convex_hull = cv2.convexHull(contours)
    # Decide the area of contours (with delta relaxation)
    roi = [max(0,min_y-delta), min(max_y+delta, frame.shape[0]), max(0,min_x-delta), min(max_x+delta,frame.shape[1])]
    return roi
