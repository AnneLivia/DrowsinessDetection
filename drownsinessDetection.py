import dlib
import cv2
import imutils
import playsound
from threading import Thread
import time
from imutils import face_utils
import argparse
from scipy.spatial import distance

EAR_THRESHOLD = 0.25
FRAMES_QTD_UNDER_THRESHOLD = 20
COUNTER = 0
ALARM = False

# command line arguments
parser = argparse.ArgumentParser('DrowsinessDetection')
parser.add_argument('-a', type=int, default=0, help="Turn on the alarm sound")
parser.add_argument('-w', type=int, default=0, help="Index of the webcam")

args = parser.parse_args();

## creating the landmark predictor model and face detector
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat');

cap = cv2.VideoCapture(args.w);

def detect_face(frame):
    # 0 is the scale factor
    return face_detector(frame, 0)

def detect_landmarks(frame, rect):
    landmarks = landmark_predictor(frame, rect)
    return face_utils.shape_to_np(landmarks)

## With FACIAL_LANDMARKS I can get the indexes for:
## ('mouth', 'inner_mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye', 'nose', 'jaw')
def get_left_eyes(landmarks):
    (start_left_eye, end_left_eye) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye'];
    return landmarks[start_left_eye:end_left_eye]

def get_right_eyes(landmarks):
    (start_right_eye, end_right_eye) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye'];
    return landmarks[start_right_eye:end_right_eye]

def draw_rect_and_convexHull(frame, rect, left_eye, right_eye):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.drawContours(frame, cv2.convexHull(left_eye), -1, (0, 0, 255), 2)
    cv2.drawContours(frame, cv2.convexHull(right_eye), -1, (0, 0, 255), 2)
    
# Computing the eye aspect ratio (EAR) between height and width of the eye
def compute_EAR(eye):
    # compute euclidian distance between vertical points
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # compute euclidian distance between horizontal points 
    C = distance.euclidean(eye[0], eye[3])

    # applying formula to calculate aspect ratio
    EAR = (A + B) / 2 * C

    return EAR

def play_alarm(path):
    playsound.playsound(path)

def drownsinessVerification(ear):
    # to use the global counter
    global COUNTER, ALARM
    print(COUNTER)
    if ear <= EAR_THRESHOLD:
        COUNTER+=1

        # if the minimum number of frame under a threshold is achieved
        if COUNTER >= FRAMES_QTD_UNDER_THRESHOLD:
            # desenha um alarme no quadro
            cv2.putText(frame, "DROWNSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if args.a == 1 and not ALARM:
                ALARM = True
                t = Thread(target=play_alarm, args=("audio/alarm.wav",))
                t.daemon = True
                t.start()
    else:
        COUNTER = 0
        ALARM = False

    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
     
while True:
    ret, frame = cap.read()
    if not ret:
        break;
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ## detecting face
    face_rects = detect_face(gray_frame)

    ## detecting landmarks
    for rect in face_rects:
        landmarks = detect_landmarks(frame, rect)

        # getting left and right eyes indexes, to calculate EAR
        left_eye = get_left_eyes(landmarks)
        right_eye = get_right_eyes(landmarks)

        
        # drawing rectangle and landmarks
        draw_rect_and_convexHull(frame, rect, left_eye, right_eye)
        
        # getting EAR
        ear_left = compute_EAR(left_eye)
        ear_right = compute_EAR(right_eye)
        
        # getting the everage EAR of both eyes (since they blink synchronously)
        avg_ear = (ear_left - ear_right) / 2

        # checking if EAR is under threshold and commpute
        drownsinessVerification(avg_ear)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

