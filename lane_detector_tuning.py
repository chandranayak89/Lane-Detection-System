import cv2
import numpy as np
import os
from improved_lane_detection import ImprovedLaneDetector

# Global variables
detector = None
frame = None
params = {}
window_name = "Lane Detection Tuning"

def create_trackbars():
    """Create trackbars for parameter tuning"""
    global params
    
    # Create window
    cv2.namedWindow(window_name)
    
    # Initial parameter values
    params = {
        'canny_low': 70,
        'canny_high': 170,
        'hough_threshold': 40,
        'min_line_length': 60,
        'max_line_gap': 30,
        'min_slope': 50,  # Multiplied by 100 for trackbar
        'max_slope': 200,  # Multiplied by 100 for trackbar
        'roi_top_width': 120,  # Width at the top of trapezoid
        'roi_bottom_width': 500,  # Width at the bottom of trapezoid
        'roi_height': 200,  # Height of ROI from bottom
    }
    
    # Create trackbars
    cv2.createTrackbar('Canny Low', window_name, params['canny_low'], 255, on_canny_low_change)
    cv2.createTrackbar('Canny High', window_name, params['canny_high'], 255, on_canny_high_change)
    cv2.createTrackbar('Hough Threshold', window_name, params['hough_threshold'], 100, on_hough_threshold_change)
    cv2.createTrackbar('Min Line Length', window_name, params['min_line_length'], 200, on_min_line_length_change)
    cv2.createTrackbar('Max Line Gap', window_name, params['max_line_gap'], 200, on_max_line_gap_change)
    cv2.createTrackbar('Min Slope (x100)', window_name, params['min_slope'], 300, on_min_slope_change)
    cv2.createTrackbar('Max Slope (x100)', window_name, params['max_slope'], 300, on_max_slope_change)
    cv2.createTrackbar('ROI Top Width', window_name, params['roi_top_width'], 640, on_roi_top_width_change)
    cv2.createTrackbar('ROI Bottom Width', window_name, params['roi_bottom_width'], 640, on_roi_bottom_width_change)
    cv2.createTrackbar('ROI Height', window_name, params['roi_height'], 480, on_roi_height_change)

def update_detector():
    """Update the detector with current parameter values"""
    global detector, params
    
    # Calculate ROI vertices based on parameters
    height, width = frame.shape[:2]
    
    # Calculate ROI vertices
    x_bottom_left = max(0, (width - params['roi_bottom_width']) // 2)
    x_bottom_right = min(width, (width + params['roi_bottom_width']) // 2)
    x_top_left = max(0, (width - params['roi_top_width']) // 2)
    x_top_right = min(width, (width + params['roi_top_width']) // 2)
    
    y_bottom = height
    y_top = max(0, height - params['roi_height'])
    
    roi_vertices = np.array([
        [(x_bottom_left, y_bottom), 
         (x_top_left, y_top), 
         (x_top_right, y_top), 
         (x_bottom_right, y_bottom)]
    ])
    
    # Update detector parameters
    detector = ImprovedLaneDetector(
        canny_low_threshold=params['canny_low'],
        canny_high_threshold=params['canny_high'],
        hough_threshold=params['hough_threshold'],
        min_line_length=params['min_line_length'],
        max_line_gap=params['max_line_gap'],
        min_slope=params['min_slope'] / 100.0,
        max_slope=params['max_slope'] / 100.0,
        roi_vertices=roi_vertices,
        smoothing_buffer_size=10
    )

# Trackbar callback functions
def on_canny_low_change(val):
    global params
    params['canny_low'] = val
    update_detector()

def on_canny_high_change(val):
    global params
    params['canny_high'] = val
    update_detector()

def on_hough_threshold_change(val):
    global params
    params['hough_threshold'] = val
    update_detector()

def on_min_line_length_change(val):
    global params
    params['min_line_length'] = val
    update_detector()

def on_max_line_gap_change(val):
    global params
    params['max_line_gap'] = val
    update_detector()

def on_min_slope_change(val):
    global params
    params['min_slope'] = val
    update_detector()

def on_max_slope_change(val):
    global params
    params['max_slope'] = val
    update_detector()

def on_roi_top_width_change(val):
    global params
    params['roi_top_width'] = val
    update_detector()

def on_roi_bottom_width_change(val):
    global params
    params['roi_bottom_width'] = val
    update_detector()

def on_roi_height_change(val):
    global params
    params['roi_height'] = val
    update_detector()

def print_current_params():
    """Print current parameter values to console"""
    print("\nCurrent Parameter Values:")
    print("-------------------------")
    print(f"canny_low_threshold={params['canny_low']}")
    print(f"canny_high_threshold={params['canny_high']}")
    print(f"hough_threshold={params['hough_threshold']}")
    print(f"min_line_length={params['min_line_length']}")
    print(f"max_line_gap={params['max_line_gap']}")
    print(f"min_slope={params['min_slope'] / 100.0}")
    print(f"max_slope={params['max_slope'] / 100.0}")
    print("\nROI Parameters:")
    print(f"roi_top_width={params['roi_top_width']}")
    print(f"roi_bottom_width={params['roi_bottom_width']}")
    print(f"roi_height={params['roi_height']}")
    print("-------------------------")

def main():
    """Main function to run the lane detection parameter tuning"""
    global detector, frame
    
    # Check if video file exists
    video_path = "road_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Please place a road video file named 'road_video.mp4' in the same directory")
        print("or update the video_path variable with the correct path to your video file.")
        return
    
    # Load video
    video = cv2.VideoCapture(video_path)
    
    # Get first frame to setup detector
    ret, frame = video.read()
    if not ret:
        print("Error reading video")
        return
    
    # Setup trackbars and detector
    create_trackbars()
    update_detector()
    
    print("Lane Detection Parameter Tuning")
    print("===============================")
    print("Instructions:")
    print("1. Adjust sliders to modify detection parameters")
    print("2. Press 'p' to print current parameter values to console")
    print("3. Press 's' to save the current frame with lane detection")
    print("4. Press 'q' to quit")
    
    # Process video
    while True:
        # Get frame (loop video if needed)
        ret, frame = video.read()
        if not ret:
            # Reset video to beginning
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = video.read()
            if not ret:
                break
        
        # Process frame with current parameters
        result = detector.process_frame(frame)
        
        # Create visualization for easier tuning
        viz = detector.visualize_steps(frame)
        viz_resized = cv2.resize(viz, (frame.shape[1], frame.shape[0]))
        
        # Show results
        cv2.imshow(window_name, result)
        cv2.imshow('Visualization', viz_resized)
        
        # Key handling
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Print current parameters
            print_current_params()
        elif key == ord('s'):  # Save current frame
            cv2.imwrite('tuned_frame.jpg', result)
            cv2.imwrite('tuned_visualization.jpg', viz_resized)
            print("Saved current frames as 'tuned_frame.jpg' and 'tuned_visualization.jpg'")
    
    # Clean up
    video.release()
    cv2.destroyAllWindows()
    
    # Print final parameters for use in your code
    print("\nFinal Parameter Values for improved_lane_detection.py:")
    print_current_params()

if __name__ == "__main__":
    main() 