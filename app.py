import cv2
import mediapipe as mp
import torch
import numpy as np
import math
import tempfile
import os
import gradio as gr

# -----------------------------
# Configuration & Initialization
# -----------------------------

# Initialize MediaPipe Pose and drawing utilities.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load your local YOLOv5 model for javelin detection.
# Adjust these paths as needed.
model = torch.hub.load('/Users/user/yolov5', 'custom',
                       path='/Users/user/yolov5/runs/train/exp/weights/best.pt',
                       source='local')
model.conf = 0.5  # detection confidence threshold


# -----------------------------
# Helper: Determine fixed orientation from first frame.
# -----------------------------
def determine_orientation(cap, pose):
    # Read the first frame.
    ret, frame = cap.read()
    if not ret:
        return "ltr"  # default if no frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    orientation = "ltr"  # default: use top-left -> bottom-right
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if left_shoulder.x < right_shoulder.x:
            orientation = "ltr"  # left-to-right
        else:
            orientation = "rtl"  # right-to-left
    # Reset video to the first frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return orientation


# -----------------------------
# Helper: Compute dominant line in ROI via Hough Transform.
# -----------------------------
def get_dominant_line_angle(roi, debug=False):
    """
    Process an ROI (BGR image) to extract edges and run HoughLinesP to get the longest line.
    Returns the angle (in degrees, normalized to [0,180]) and the line coordinates.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if debug:
        cv2.imshow('ROI Edges', edges)
        cv2.waitKey(1)

    diag = math.hypot(roi.shape[0], roi.shape[1])
    min_line_length = int(0.4 * diag)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=25,
                            minLineLength=min_line_length, maxLineGap=15)
    if lines is None:
        return None, None

    longest_length = 0
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > longest_length:
            longest_length = length
            best_line = (x1, y1, x2, y2)
    if best_line is None:
        return None, None

    # Calculate the angle relative to horizontal.
    x1, y1, x2, y2 = best_line
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = abs(math.degrees(angle_rad)) % 180
    return angle_deg, best_line


# -----------------------------
# Main Processing Function with Fixed Orientation and Side Display
# -----------------------------
def process_video(video_file):
    """
    Processes each frame:
      - Runs MediaPipe Pose detection and draws the pose.
      - Runs YOLOv5 detection for the javelin.
      - Draws a rectangle around the detected javelin.
      - Based on a fixed decision from the first frame, draws a line across the detection box.
      - Displays the line's angle above the box and on the left side of the frame.
    Returns the path to the output video.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return "Error: Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary output file.
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_output.name
    temp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize MediaPipe Pose.
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        # Decide fixed orientation based on the first frame.
        orientation = determine_orientation(cap, pose)
        print(f"Fixed orientation: {orientation}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            annotated_frame = frame.copy()

            # Draw pose landmarks.
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_frame,
                                          pose_results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS)

            # Run YOLOv5 detection for the javelin.
            results = model([rgb_frame])
            detections = results.xyxy[0].cpu().numpy()

            if len(detections) > 0:
                # Use the highest confidence detection.
                detections = sorted(detections, key=lambda x: -x[4])
                det = detections[0]
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Define the four corners of the bounding box.
                top_left = (x1, y1)
                bottom_right = (x2, y2)
                bottom_left = (x1, y2)
                top_right = (x2, y1)

                # Decide which diagonal to use based on the fixed orientation.
                if orientation == "ltr":
                    line_start, line_end = top_left, bottom_right
                else:
                    line_start, line_end = bottom_left, top_right

                # Draw the javelin line (with increased thickness if desired).
                cv2.line(annotated_frame, line_start, line_end, (0, 0, 255), 4)

                # Calculate the angle of the line.
                angle_rad = math.atan2(line_end[1] - line_start[1], line_end[0] - line_start[0])
                angle_deg = abs(math.degrees(angle_rad)) % 180

                # Display the angle above the box.
                cv2.putText(annotated_frame, f"Angle: {angle_deg:.1f} deg",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Also display the angle on the left side of the frame.
                cv2.putText(annotated_frame, f"Angle: {angle_deg:.1f} deg",
                            (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No javelin detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            writer.write(annotated_frame)

    cap.release()
    writer.release()
    return output_path


# -----------------------------
# Create Gradio Interface -
# -----------------------------
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Javelin Detector with Pose & Angle",
    description=("Upload a video to see pose landmarks, a javelin detection box, and a "
                 "diagonal line drawn across the box with its angle (displayed above the box and on the side).")
)

if __name__ == '__main__':
    iface.launch()
