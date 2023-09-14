from dataclasses import dataclass
from enum import IntEnum
from typing import List

class LANDMARKERID(IntEnum):
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    ELBOW_LEFT = 13
    ELBOW_RIGHT = 14
    WRIST_LEFT = 15
    WRIST_RIGHT = 16
    INDEX_FINGER_LEFT = 19
    INDEX_FINGER_RIGHT = 20
    HIP_LEFT = 23
    HIP_RIGHT = 24
    KNEE_LEFT = 25
    KNEE_RIGHT = 26
    ANKLE_LEFT = 27
    ANKLE_RIGHT = 28

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
    image: any
    landmarkers: List[Landmarker]
    angles: List[Angle]