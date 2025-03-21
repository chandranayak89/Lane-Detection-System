import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_frame(frame):
    """
    Convert the frame to grayscale and apply Gaussian blur
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(frame):
    """
    Apply Canny edge detection
    """
    return cv2.Canny(frame, 50, 150)

def region_of_interest(image):
    """
    Define a region of interest (ROI) where lane lines are likely to be
    """
    height, width = image.shape
    mask = np.zeros_like(image)

    # Define a polygon mask (trapezoidal ROI)
    # This can be adjusted based on camera position and view
    polygon = np.array([
        [(50, height), (width - 50, height), (width // 2, height // 2)]
    ])
    
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def detect_lines(image):
    """
    Use Hough Transform to detect lines in the edge-detected image
    """
    return cv2.HoughLinesP(image, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

def draw_lines(image, lines):
    """
    Draw detected lines on the original frame
    """
    if lines is None:
        return image
    line_image = np.zeros_like(image)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

def lane_detection_pipeline(frame):
    """
    Combine all steps for lane detection
    """
    processed = preprocess_frame(frame)
    edges = detect_edges(processed)
    roi = region_of_interest(edges)
    lines = detect_lines(roi)
    return draw_lines(frame, lines)

def main():
    # Check if video file exists
    video_path = "road_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Please place a road video file named 'road_video.mp4' in the same directory")
        print("or update the video_path variable with the correct path to your video file.")
        return

    # Load the video file
    video = cv2.VideoCapture(video_path)

    # Create output video writer (optional)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = "lane_detection_output.avi"
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("End of video stream or error reading video")
            break

        # Process the frame through our lane detection pipeline
        lane_frame = lane_detection_pipeline(frame)
        
        # Write to output video
        output.write(lane_frame)
        
        # Display the processed frame
        cv2.imshow("Lane Detection", lane_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Clean up
    video.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 