from dataclasses import dataclass
import math
from typing import List
import cv2
from draw_util import draw_angle
from dtypes import LANDMARKERID, Angle, Landmarker

def get_landmarks_list(img, results):
    if results.pose_landmarks:
        landmark_list = []
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            # finding height, width of the image printed
            height, width, c = img.shape
            # Determining the pixels of the landmarks
            landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(
                landmark.y * height
            )
            landmark_list.append(Landmarker(id, landmark_pixel_x, landmark_pixel_y))
        return landmark_list
    raise ValueError()

def find_knee_height(
    landmark_list
):
    # Retrieve landmark coordinates from point identifiers
    _, y1 = landmark_list[LANDMARKERID.KNEE_LEFT][1:]
    _, y2 = landmark_list[LANDMARKERID.KNEE_RIGHT][1:]

    return (y1+y2)/2

def find_lowest_knee(
    list_results
):
    # Retrieve landmark coordinates from point identifiers
    min_y = 100000
    min_img = None
    for landmarks_list, image in list_results:
        y = find_knee_height(landmarks_list)
        if y < min_y:
            min_y = y
            min_img = image
    print(min_y)

def calculate_distance(
    pointA: Landmarker, pointB: Landmarker
):
    return math.sqrt((pointB.x-pointA.x)**2+(pointB.y-pointA.y)**2)

def get_angle_from_landmarkers(
    point1: Landmarker, point2: Landmarker, point3: Landmarker
) -> Angle:
    # Retrieve landmark coordinates from point identifiers
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    x3, y3 = point3.x, point3.y

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Handling angle edge cases: Obtuse and negative angles
    if angle < 0:
        angle += 360
        if angle > 180:
            angle = 360 - angle
    elif angle > 180:
        angle = 360 - angle
    return Angle(point1, point2, point3, angle)

    
def _get_angle(landmark_list: List[Landmarker], id1:LANDMARKERID, id2:LANDMARKERID.ELBOW_RIGHT, id3:LANDMARKERID.WRIST_RIGHT) -> Angle:
    return get_angle_from_landmarkers(landmark_list[id1], landmark_list[id2], landmark_list[id3])

def get_elbow_right_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.SHOULDER_RIGHT, LANDMARKERID.ELBOW_RIGHT, LANDMARKERID.WRIST_RIGHT)

def get_elbow_left_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.SHOULDER_LEFT, LANDMARKERID.ELBOW_LEFT, LANDMARKERID.WRIST_LEFT)

def get_shoulder_right_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.ELBOW_RIGHT, LANDMARKERID.SHOULDER_RIGHT, LANDMARKERID.HIP_RIGHT)

def get_shoulder_left_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.ELBOW_LEFT, LANDMARKERID.SHOULDER_RIGHT, LANDMARKERID.HIP_RIGHT)

def get_hip_right_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.SHOULDER_RIGHT, LANDMARKERID.HIP_RIGHT, LANDMARKERID.KNEE_RIGHT)

def get_hip_left_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.SHOULDER_LEFT, LANDMARKERID.HIP_LEFT, LANDMARKERID.KNEE_LEFT)

def get_knee_right_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.HIP_RIGHT, LANDMARKERID.KNEE_RIGHT, LANDMARKERID.ANKLE_RIGHT)#

def get_knee_left_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.HIP_LEFT, LANDMARKERID.KNEE_LEFT, LANDMARKERID.ANKLE_LEFT)

def get_wrist_right_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.ELBOW_RIGHT, LANDMARKERID.WRIST_RIGHT, LANDMARKERID.INDEX_FINGER_RIGHT)

def get_wrist_left_angle(landmark_list):
    return _get_angle(landmark_list, LANDMARKERID.ELBOW_LEFT, LANDMARKERID.WRIST_LEFT, LANDMARKERID.LEFT_THUMB)
