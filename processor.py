from typing import List
from draw_util import draw_angle
from dtypes import Frame, Landmarker
from util import get_elbow_right_angle, get_knee_right_angle


class FrameProcessor:
    frames: List[Frame] = []

    def processPose(self, image, landmarks: List[Landmarker]):
        angles = [
            get_elbow_right_angle(landmarks),
            get_knee_right_angle(landmarks)
        ]
        for angle in angles:
            draw_angle(image, angle)
            self.frames.append(Frame(image, landmarks, angles))
        