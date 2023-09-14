import mediapipe as mp
import cv2
import sys
from processor import FrameProcessor

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2, model_complexity=1) 
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
frameProcessor = FrameProcessor(pose)
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)         
    image.flags.writeable = False
    try:
        frameProcessor.processImage(image)
    except:
        continue

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      

    out.write(image)

pose.close()
cap.release()
out.release()

