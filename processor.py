from typing import List

import cv2
from draw_util import draw_angle
from dtypes import LANDMARKERID, Frame, Landmarker
from util import calculate_distance, get_ankle_left_angle, get_elbow_left_angle, get_elbow_right_angle, get_hip_left_angle, get_hip_right_angle, get_knee_left_angle, get_knee_right_angle, get_landmarks_list, get_shoulder_left_angle, get_shoulder_right_angle, get_wrist_left_angle, get_wrist_right_angle
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

class FrameProcessor:
    frames: List[Frame] = []
    pose: mp.solutions.pose.Pose
    frame_counter = 0

    def __init__(self, pose: mp.solutions.pose.Pose):
        self.pose = pose

    def find_release_frame(self):
        #ReleaseFrame = When Wrist is furthest from head and higher than head
        min_y = 10000
        ERROR_MARGIN = 50
        max_frame = None
        for frame in self.frames:
            wristLeft = frame.landmarkers[LANDMARKERID.WRIST_LEFT]
            nose = frame.landmarkers[LANDMARKERID.NOSE]
            if wristLeft.y < nose.y:
                if wristLeft.y <= min_y-ERROR_MARGIN:
                    min_y = wristLeft.y
                    max_frame = frame
        return max_frame
    
    def find_shot_start_frame(self):
        #StartFrame = When knees are lowest and before release
        max_y = 0
        ERROR_MARGIN = 10
        max_frame = None
        for frame in self.frames:
            kneeLeft = frame.landmarkers[LANDMARKERID.WRIST_LEFT]
            if kneeLeft.y >= max_y+ERROR_MARGIN:
                max_y = kneeLeft.y
                max_frame = frame
        return max_frame
    
    def find_knees_fully_released_frame(self, shot_start_frame: Frame):
        #StartFrame = When knees are lowest and before release
        max_angle = 0
        ERROR_MARGIN = 5
        max_frame = None
        for frame in self.frames:
            knee_angle = frame.angles[1].angle #TODO name better
            if frame.number>= shot_start_frame.number and knee_angle >= max_angle+ERROR_MARGIN:
                max_angle = knee_angle
                max_frame = frame
        return max_frame
    
    def find_jump_frame(self, shot_start_frame: Frame):
        #StartFrame = When angle of ankle is biggest after shot start
        max_angle = 0
        ERROR_MARGIN = 5
        max_frame = None
        for frame in self.frames:
            ankle_angle = frame.angles[5].angle #TODO name better
            if frame.number>= shot_start_frame.number and ankle_angle >= max_angle+ERROR_MARGIN:
                max_angle = ankle_angle
                max_frame = frame
        return max_frame


    def process_image(self, image):
        results = self.pose.process(image)
        try:
            landmarks_list = get_landmarks_list(image, results)
        except Exception as e:
            print("Can't process frame, no points")
            raise e
        
        angles = [
            get_elbow_left_angle(landmarks_list),
            get_knee_left_angle(landmarks_list),
            get_wrist_left_angle(landmarks_list),
            get_hip_left_angle(landmarks_list),
            get_shoulder_left_angle(landmarks_list),
            get_ankle_left_angle(landmarks_list)
        ]
        for angle in angles:
            draw_angle(image, angle)
            self.frames.append(Frame(self.frame_counter, image, image, landmarks_list, angles))
            self.frame_counter += 1

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    
    def post_process(self):
        release_frame = self.find_release_frame()
        release_frame.pause = True
        cv2.putText(release_frame.image, str("RELEASE"), (50, 500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2) 

        shot_start_frame = self.find_shot_start_frame()
        shot_start_frame.pause = True
        cv2.putText(shot_start_frame.image, str("SHOT Start"), (50, 500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2) 

        knees_released_frame = self.find_knees_fully_released_frame(shot_start_frame)
        knees_released_frame.pause = True
        cv2.putText(knees_released_frame.image, str("Knees Fully Extended"), (50, 500), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2) 

        jump_frame = self.find_jump_frame(shot_start_frame)
        jump_frame.pause = True
        cv2.putText(jump_frame.image, str("JUMP Start"), (50, 600), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2) 
