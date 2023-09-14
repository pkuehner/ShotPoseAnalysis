from typing import List
from draw_util import draw_angle
from dtypes import Frame, Landmarker
from util import get_elbow_left_angle, get_elbow_right_angle, get_hip_left_angle, get_hip_right_angle, get_knee_left_angle, get_knee_right_angle, get_landmarks_list, get_shoulder_left_angle, get_shoulder_right_angle, get_wrist_left_angle, get_wrist_right_angle
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

class FrameProcessor:
    frames: List[Frame] = []
    pose: mp.solutions.pose.Pose

    def __init__(self, pose: mp.solutions.pose.Pose):
        self.pose = pose

    def processImage(self, image):
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
        ]
        for angle in angles:
            draw_angle(image, angle)
            self.frames.append(Frame(image, landmarks_list, angles))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        