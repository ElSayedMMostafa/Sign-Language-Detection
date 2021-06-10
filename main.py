#this script works with OpenCV4
import cv2
from helpers import *

#Start Live Camera
cap = cv2.VideoCapture(0)

#Get Region of Interest
success, img = cap.read()
roi = segment_hand(img,0)

#Define Boundry Box
bbox = (roi[2], roi[0], roi[3]-roi[2], roi[1]-roi[0])

#Initialize Tracker
#tracker = cv2.TrackerCSRT_create()
tracker = cv2.TrackerMOSSE_create()
print('Begginning detection')
success = tracker.init(img, bbox)
print('Start Tracking')

count = 0
while True:
    #Get current Frame
    success, img = cap.read()
    
    #count frames 
    count +=1
    n_frame_update = 60 #number of frames after which run the detection and reset the tracker

    #Run detection every n_frame_update
    if count%n_frame_update == 0:

        #Run detection 
        roi = segment_hand(img,0)
        bbox = (roi[2], roi[0], roi[3]-roi[2], roi[1]-roi[0])
        
        #delete old tracker and create a new one
        del tracker
        tracker = cv2.TrackerMOSSE_create()
        success = tracker.init(img, bbox)

        #print on screen 
        cv2.putText(img, "Detecting", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)
        
        count = 0 #to avoid over flow
    
    else:
        # Update tracker
        success, bbox = tracker.update(img)
    
    #if Tracker succeded
    if success:

        #Draw Boundry Box
        start_point = (int(bbox[0]),int(bbox[1]))
        end_point = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
        cv2.rectangle(img, start_point, end_point, (255,0,0), 2)

    else:
        #If Tracker Failed
        cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    #Show image
    cv2.imshow("Tracking", img)
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break