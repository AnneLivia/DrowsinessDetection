import dlib 
import cv2
import argparse
from imutils import face_utils
from scipy.spatial import distance
# library used to play a sound
import pygame

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('audio/alarm.wav')

# to control alarm playing, otherwise, the method to play a sound is going to be called multiple times
PLAYED_ALARM = False

# video dimensions
VIDEO_HEIGHT = 520
VIDEO_WIDTH = 700

# defining arguments
parser = argparse.ArgumentParser()
parser.add_argument('-wi', type=int, default=0, help='The index of the webcam on your system')
parser.add_argument('-a', type=int, default=0, help='Turn on the alarm sound by passing 1')
parser.add_argument('-vi', help='The path to a video you want to analyze')

args = parser.parse_args()

# variables to handle drownsiness verification
EAR_THRESHOLD = 0.20;
FRAMES_COUNTER = 0
PLAY_ALARM_SOUND = True if args.a == 1 else False
QTD_FRAME_UNDER_THRESHOLD = 15;

# trackbar created to control ear threshold value
cv2.namedWindow('Video')
def control_ear_threshold(e):
    global EAR_THRESHOLD
    EAR_THRESHOLD = e / 100.00

cv2.createTrackbar('THRESHOLD', 'Video', 20, 100, control_ear_threshold)

# loading predictors and face detector
landmark_predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

# load a video if it was passed as an argument otherwise it load the webcam
cap = cv2.VideoCapture(args.vi if args.vi else args.wi)

def play_alarm_sound(start = True):
    # use global when change a value
    global PLAYED_ALARM
    if (start):
        pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.stop()
        # to allow to play the alarm again
        PLAYED_ALARM = False

def detect_face(frame):
    return face_detector(frame)

def detect_landmarks(frame, rect):
    shape = landmark_predictor(frame, rect)
    return face_utils.shape_to_np(shape)

# from all the landmarks obtained, get only left and right eye points
def get_left_and_right_eye_points(landmarks):
    # with FACIAL_LANDMARKS_68_IDXS, we can get:
    # 'mouth', 'inner_mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye', 'nose', 'jaw'
    (start_left_eye, end_left_eye) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
    (start_right_eye, end_right_eye) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

    return landmarks[start_left_eye:end_left_eye], landmarks[start_right_eye:end_right_eye]

# EAR is the eye aspect ratio, and this is computed to check for opening and closing the eyes
# to compute the EAR, we need to compute the distance between P2 and P6, P3 and P5 and, P1 and P4
# P2, P6, P3 and P5 represents the height of the eye, whereas P1 and P4 represents the width.
# After compute the distance between all of these points, we need to average the result
# because the blink is done considering both eyes simultaneosly. More information about this computation 
# can be found in this paper: https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def compute_ear_avg(left_eye, right_eye):
    # euclidean distance
    height_1_left = distance.euclidean(left_eye[1], left_eye[5])
    height_2_left = distance.euclidean(left_eye[2], left_eye[4])
    width_left = distance.euclidean(left_eye[0], left_eye[3])

    height_1_right = distance.euclidean(right_eye[1], right_eye[5])
    height_2_right = distance.euclidean(right_eye[2], right_eye[4])
    width_right = distance.euclidean(right_eye[0], left_eye[3])

    ear_left = (height_1_left + height_2_left) / (2 * width_left)
    ear_right = (height_1_right + height_2_right) / (2 * width_right)

    return (ear_left + ear_right) / 2

# if the ear computed is bellow the threshold set, then the person is with the eye almost or exactly closed
def detect_drownsiness(ear):
    global FRAMES_COUNTER, PLAYED_ALARM
    # determining if eyes are closing
    if ear < EAR_THRESHOLD:
        FRAMES_COUNTER+=1
        # if the minimum number of allowed frame under a threshold 
        #  was achieved then drownsiness was detected
        if FRAMES_COUNTER >= QTD_FRAME_UNDER_THRESHOLD:
            cv2.putText(
                img=resized_frame,
                text='DROWNSINESS DETECTED', 
                org=(int(VIDEO_WIDTH / 4.2), 40), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255), 
                thickness=2
            )
            
            if not PLAYED_ALARM and PLAY_ALARM_SOUND:
                play_alarm_sound()
                PLAYED_ALARM = True
    else:
        # Person opened the eyes
        FRAMES_COUNTER = 0
        play_alarm_sound(False)

    cv2.putText(
        img=resized_frame,
        text='{:.2f} EAR'.format(ear), 
        org=(int(VIDEO_WIDTH / 2.4), 80), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 255), 
        thickness=2
    )
    
def draw_rectangle_and_eyes_contour(frame, rect, left_eye, right_eye):
    x, y, w, h = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    left_hull = cv2.convexHull(left_eye)
    right_hull = cv2.convexHull(right_eye)
    
    # to draw the contour instead of each point individually
    cv2.drawContours(frame, [left_hull], -1, (0, 255, 255), 2)
    cv2.drawContours(frame, [right_hull], -1, (0, 255, 255), 2)


while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    resized_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # get the rectangle where the face is
    face_rects = detect_face(gray)

    # in case, a face is not detected and the alarm is still playing
    if not face_rects:
        # Person opened the eyes
        FRAMES_COUNTER = 0
        # to allow to play the alarm again
        play_alarm_sound(False)

    for rect in face_rects:
        # get the face landmarks considering the rect of the face
        landmarks = detect_landmarks(gray, rect)

        # getting left and right eyes points, to compute the EAR
        left_eye, right_eye = get_left_and_right_eye_points(landmarks)

        # compute the EAR
        ear = compute_ear_avg(left_eye, right_eye)
        
        print('EAR: {}'.format(ear))

        # drawing all the necessary rectangles and points on the face
        draw_rectangle_and_eyes_contour(resized_frame, rect, left_eye, right_eye)

        detect_drownsiness(ear)

    cv2.imshow('Video', resized_frame)

    if cv2.waitKey(1) >= 0:
        break;

cap.release()
cv2.destroyAllWindows()




