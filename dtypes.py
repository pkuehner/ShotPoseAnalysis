from dataclasses import dataclass
from enum import IntEnum
from typing import List


class LANDMARKERID(IntEnum):
    NOSE = 0
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    ELBOW_LEFT = 13
    ELBOW_RIGHT = 14
    WRIST_LEFT = 15
    WRIST_RIGHT = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    INDEX_FINGER_LEFT = 19
    INDEX_FINGER_RIGHT = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    HIP_LEFT = 23
    HIP_RIGHT = 24
    KNEE_LEFT = 25
    KNEE_RIGHT = 26
    ANKLE_LEFT = 27
    ANKLE_RIGHT = 28
    HEEL_LEFT = 29
    HEEL_RIGHT = 30
    FOOT_INDEX_LEFT = 31
    FOOT_INDEX_RIGHT = 31


@dataclass
class Landmarker:
    id: LANDMARKERID
    x: int
    y: int


@dataclass
class Angle:
    landmarker1: Landmarker
    landmarker2: Landmarker
    landmarker3: Landmarker
    angle: float


@dataclass
class Frame:
    number: int
    image: any
    landmarkers: List[Landmarker]
    angles: List[Angle]
    pause: bool = False
