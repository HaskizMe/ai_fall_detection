# # This version works pretty good
# from transformers import pipeline
# import cv2
# import os

# # Initialize the video classification pipeline
# pipe = pipeline("video-classification", model="yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect")

# # Define the path to your video file
# video_path = "fall.mp4"
# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count / fps
# segment_duration = 2  # seconds per segment

# fall_predictions = []

# # Directory to store temporary sub-videos
# os.makedirs("temp_segments", exist_ok=True)

# # Process video in segments and track timestamps
# for start_time in range(0, int(duration), segment_duration):
#     cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

#     # Create a VideoWriter to save each sub-video segment
#     segment_path = f"temp_segments/segment_{start_time}.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(segment_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

#     for _ in range(int(fps * segment_duration)):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)

#     out.release()

#     # Run the model on each sub-video
#     result = pipe(segment_path)

#     # Log only fall predictions with timestamps
#     for pred in result:
#         if 'fall' in pred['label'].lower():  # Adjust the condition as needed
#             fall_predictions.append({
#                 "time": start_time,
#                 "label": pred["label"],
#                 "confidence": pred["score"]
#             })
#             print(f"FALL DETECTED! Time: {start_time} sec - Label: {pred['label']}, Confidence: {pred['score']}")

#     # Remove the temporary segment video
#     # os.remove(segment_path)

# cap.release()

# # Remove the temporary directory if empty
# os.rmdir("temp_segments")

# # Optional: Print all fall predictions at once
# if fall_predictions:
#     print("\nAll fall predictions:")
#     print(f"All Falls: {len(fall_predictions)}")
#     for fall in fall_predictions:
#         print(f"Time: {fall['time']} sec - Label: {fall['label']}, Confidence: {fall['confidence']}")
# else:
#     print("No falls detected.")



# Simple one just using hugging face model
# from transformers import pipeline

# # Initialize the video classification pipeline
# pipe = pipeline("video-classification", model="yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect")

# # Define the path to your video file
# video_path = "fall2.mp4"

# # Use the pipeline to classify the video
# results = pipe(video_path)

# # Output the predictions
# for result in results:
#     if(result['label'] == "FallDown"):
#         print(f"Predicted label: {result['label']}, Confidence: {result['score']}")



# import cv2
# import numpy as np
# from ultralytics import YOLO
# import cvzone
# import collections
# import os
# import datetime

# # Initialize YOLOv8 model
# model = YOLO("yolo11s.pt")
# names = model.model.names

# # Open the webcam (0 for default webcam)
# cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# # Variables for recording
# recording = False
# video_writer = None
# output_directory = "fall_recordings"  # Save to the fall_recordings directory

# # Ensure the output directory exists
# os.makedirs(output_directory, exist_ok=True)

# pre_fall_buffer = collections.deque(maxlen=int(fps * 3))  # 3-second buffer before fall detection
# post_fall_duration = 8  # Record 8 seconds after fall detection
# frame_counter = 0  # To track frames post fall
# fall_detected = False  # Keep track of fall status

# # Function to start recording
# def start_recording(frame_size, fps):
#     global video_writer
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = os.path.join(output_directory, f"fall_{timestamp}.mp4")
#     video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
#     print(f"Started recording: {filename}")

# # Function to stop recording
# def stop_recording():
#     global video_writer
#     if video_writer is not None:
#         video_writer.release()
#         video_writer = None
#         print("Stopped recording.")

# # Main processing loop
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.resize(frame, (1020, 600))

#     # Add frame to pre-fall buffer (for the rolling 3-second window)
#     pre_fall_buffer.append(frame.copy())

#     # Run YOLOv8 tracking on the frame
#     results = model.track(frame, persist=True, classes=0)

#     fall_detected_current_frame = False
#     if results[0].boxes is not None and results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.int().cpu().tolist()
#         class_ids = results[0].boxes.cls.int().cpu().tolist()
#         track_ids = results[0].boxes.id.int().cpu().tolist()
#         confidences = results[0].boxes.conf.cpu().tolist()

#         for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
#             x1, y1, x2, y2 = box
#             h = y2 - y1
#             w = x2 - x1
#             thresh = h - w
#             print(thresh)

#             if thresh <= 0:  # Fall detected
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
#                 cvzone.putTextRect(frame, f"{'Fall'}", (x1, y1), 1, 1)
#                 fall_detected_current_frame = True
#             else:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
#                 cvzone.putTextRect(frame, f"{'Normal'}", (x1, y1), 1, 1)

#     # Start recording when fall is detected and it's not already recording
#     if fall_detected_current_frame and not fall_detected:
#         start_recording((1020, 600), fps)
#         fall_detected = True
#         frame_counter = 0  # Reset frame counter
#         # Write the buffered pre-fall frames to the video
#         for buffered_frame in pre_fall_buffer:
#             video_writer.write(buffered_frame)

#     # If currently recording (post-fall), continue recording for 8 seconds (8 * fps)
#     if fall_detected:
#         video_writer.write(frame)
#         frame_counter += 1
#         if frame_counter > fps * post_fall_duration:  # 8 seconds after the fall
#             stop_recording()
#             fall_detected = False  # Reset fall detection status

#     # Display the frame
#     cv2.imshow("RGB", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Ensure the video file is saved properly if still recording
# if fall_detected:
#     stop_recording()

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()





# # Ask user if they want to use live camera or upload a video
# print("Choose input method:")
# print("1. Live Camera")
# print("2. Video Path")
# choice = input("Enter 1 for Live Camera or 2 for Video Path: ")

# # Initialize capture source based on user choice
# if choice == "1":
#     capture_source = 0  # Use default webcam
# elif choice == "2":
#     video_path = input("Please enter the full path to the video file: ")
#     while not os.path.isfile(video_path):
#         print("Invalid path. Please try again.")
#         video_path = input("Please enter the full path to the video file: ")
#     capture_source = video_path
# else:
#     print("Invalid choice. Exiting.")
#     exit()

# Set up video capture
#cap = cv2.VideoCapture(capture_source)


# import cv2
# import numpy as np
# import os
# import datetime
# from ultralytics import YOLO
# from transformers import pipeline
# import collections

# # Initialize models
# yolo_model = YOLO("yolo11s.pt")  # YOLO fall detection model
# yolo_model(verbose = False)[0]
# video_classifier = pipeline("video-classification", model="yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect")

# # Directory to save recorded falls
# output_directory = "fall_recordings"
# os.makedirs(output_directory, exist_ok=True)

# # Set up video capture
# cap = cv2.VideoCapture("fall.mp4")  # Replace with 0 for live webcam
# fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 FPS if it can't be determined
# pre_fall_buffer = collections.deque(maxlen=int(fps * 3))  # 3 seconds of pre-fall frames
# post_fall_duration = 7  # Record 7 seconds after fall detection
# fall_frame_threshold = 50  # Fall must be detected for 10 frames
# fall_frame_count = 0
# recording = False
# video_writer = None
# fall_detected = False
# frame_counter = 0  # Frame counter for post-fall recording

# # Function to start recording
# def start_recording():
#     global video_writer, recording
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     video_path = os.path.join(output_directory, f"fall_{timestamp}.mp4")
#     video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
#     print(f"Started recording: {video_path}")
#     for buffered_frame in pre_fall_buffer:
#         video_writer.write(buffered_frame)  # Write pre-fall buffer to video
#     recording = True
#     return video_path  # Return path for analysis

# # Function to stop recording
# def stop_recording():
#     global video_writer, recording
#     if video_writer:
#         video_writer.release()
#         video_writer = None
#         recording = False
#         print("Stopped recording.")

# # Main loop to process frames
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     pre_fall_buffer.append(frame.copy())  # Add frame to pre-fall buffer
#     results = yolo_model.track(frame, persist=True, classes=0)  # YOLO detection on frame

#     # Check YOLO results for potential falls
#     fall_detected_current_frame = False
#     if results[0].boxes is not None and results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.int().cpu().tolist()
#         class_ids = results[0].boxes.cls.int().cpu().tolist()

#         # Loop through detections
#         for box, class_id in zip(boxes, class_ids):
#             x1, y1, x2, y2 = box
#             h, w = y2 - y1, x2 - x1
#             if h - w <= 0:  # Condition for fall detection
#                 fall_detected_current_frame = True
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, "Fall", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 break

#     # Check if a fall has been detected for 10 consecutive frames
#     if fall_detected_current_frame:
#         fall_frame_count += 1
#     else:
#         fall_frame_count = 0  # Reset if fall is not detected in the current frame

#     # Start recording if fall is sustained for the threshold duration
#     if fall_frame_count >= fall_frame_threshold and not fall_detected:
#         video_path = start_recording()
#         fall_detected = True
#         frame_counter = 0

#     # Continue recording for post-fall duration
#     if fall_detected:
#         video_writer.write(frame)
#         frame_counter += 1
#         if frame_counter > fps * post_fall_duration:
#             stop_recording()
#             fall_detected = False  # Reset fall detection status
#             fall_frame_count = 0  # Reset sustained fall frame count

#             # Analyze recorded video with secondary model
#             print("Analyzing recorded video for confirmation...")
#             # results = video_classifier(video_path)
#             # for pred in results:
#             #     if 'fall' in pred['label'].lower():
#             #         print(f"FALL CONFIRMED in video: {video_path}, Confidence: {pred['score']}")
#             #         break
#             # else:
#             #     print("No fall detected in recorded video.")

#     # Display the frame
#     cv2.imshow("Fall Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Ensure recording is stopped before exiting
# if recording:
#     stop_recording()
# cap.release()
# cv2.destroyAllWindows()





# This is working pretty well
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os
import numpy as np

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# Load the YOLOv8 model
model = YOLO("yolo11s.pt")
model(verbose = False)[0]
names=model.model.names
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('fall.mp4')
output_directory = "fall_recordings"
os.makedirs(output_directory, exist_ok=True)

fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 FPS if it can't be determined
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

recording = False
video_writer = None
fall_count = 0

count=0
total_falls = 0


# Function to start recording
def start_recording():
    global video_writer, recording
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = os.path.join(output_directory, f"fall_{timestamp}.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    recording = True
    print(f"Started recording: {video_path}")

# Function to stop recording
def stop_recording():
    global video_writer, recording
    if video_writer:
        video_writer.release()
        video_writer = None
        recording = False
        print("Stopped recording.")

while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,classes=0)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            h=y2-y1
            w=x2-x1
            thresh=h-w
            # print(thresh)
            
            if thresh <= 0:
                print("fall detected")
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                cvzone.putTextRect(frame,f"{'Fall'}",(x1,y1),1,1)
                fall_count = fall_count + 1

                
            else:
                print("no fall")
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                cvzone.putTextRect(frame,f"{'Normal'}",(x1,y1),1,1)
                if(fall_count > 2):
                    print(f"Total fall count: {fall_count}")
                    total_falls = total_falls + 1
                fall_count = 0

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break
# After loop ends, check if there was a sustained fall at the end
if fall_count > 2:
    total_falls += 1
    print("Fall detected at the end of video.")
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
print(f"Total falls: {total_falls}")
