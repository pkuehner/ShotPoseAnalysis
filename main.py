import mediapipe as mp
import cv2
import sys
from processor import FrameProcessor
from viz_util import visualize

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.1,
                                       category_allowlist=["sports ball", "person"],
                                       max_results=2,
                                       running_mode=mp.tasks.vision.RunningMode.VIDEO)
detector = vision.ObjectDetector.create_from_options(options)

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
fps = 90
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
frameProcessor = FrameProcessor(pose)
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)         
    image.flags.writeable = False
    try:
        frameProcessor.process_image(image)
    except:
        continue
frameProcessor.post_process()
pause_time_seconds = 5
for frame in frameProcessor.frames:
    image = frame.image
    frame_timestamp_ms = int(1000 * frame.number / fps)
    detection_result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.original_image), frame_timestamp_ms)
    annotated_image = visualize(frame.image, detection_result)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
    
    repeat_count=1
    if frame.pause:
        repeat_count = pause_time_seconds*fps
    for _ in range(repeat_count):
        out.write(image)

pose.close()
cap.release()
out.release()

