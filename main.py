import mediapipe as mp
import cv2
import sys
from processor import FrameProcessor

from util import get_landmarks_list

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2) 
print(pose) 
sys.argv.append("IMG_4290.MOV")
cap = cv2.VideoCapture(sys.argv[1])
if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind('/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_annotated.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
frameProcessor = FrameProcessor()
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)         
    image.flags.writeable = False
    results = pose.process(image)
    try:
        landmarks_list = get_landmarks_list(image, results)
    except:
        print("Can't process frame, no points")
        continue
    frameProcessor.processPose(image, landmarks_list)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    out.write(image)

pose.close()
cap.release()
out.release()

