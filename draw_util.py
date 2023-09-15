import cv2
from dtypes import Angle, Landmarker

def _draw_line(image, point1: Landmarker, point2: Landmarker):
    cv2.line(image, (point1.x, point1.y), (point2.x, point2.y), (255,255,255), 3)

def _draw_circles(image, point: Landmarker):
    cv2.circle(image, (point.x, point.y), 5, (75,0,130), cv2.FILLED)
    cv2.circle(image, (point.x, point.y), 15, (75,0,130), 2)

def draw_angle(image, angle:Angle):
    #Drawing lines between the three points
    _draw_line(image, angle.landmarker1, angle.landmarker2)
    _draw_line(image, angle.landmarker3, angle.landmarker2)

    #Drawing circles at intersection points of lines
    _draw_circles(image, angle.landmarker1)
    _draw_circles(image, angle.landmarker2)
    _draw_circles(image, angle.landmarker3)
   
    #Show angles between lines
    cv2.putText(image, str(int(angle.angle)), (angle.landmarker2.x-50, angle.landmarker2.y+50), 
    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)