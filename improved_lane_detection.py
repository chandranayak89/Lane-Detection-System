import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

class ImprovedLaneDetector:
    def __init__(self, 
                 canny_low_threshold=70,  # Increased threshold to reduce noise
                 canny_high_threshold=170, # Increased for stronger edges only
                 hough_threshold=40,      # Increased to require more votes
                 min_line_length=60,      # Longer lines to avoid small segments
                 max_line_gap=30,         # Allow gaps in lane lines
                 roi_vertices=None,
                 smoothing_buffer_size=15, # Larger buffer for more stable lines
                 min_slope=0.5,           # Minimum slope to be considered a lane line
                 max_slope=2.0,           # Maximum slope to filter out near-vertical lines
                 lane_width_px=None):     # Expected lane width in pixels (measured at bottom)
        """
        Initialize the improved lane detector with configurable parameters
        """
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.roi_vertices = roi_vertices
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.lane_width_px = lane_width_px
        
        # Buffers for smoothing
        self.left_lines_buffer = deque(maxlen=smoothing_buffer_size)
        self.right_lines_buffer = deque(maxlen=smoothing_buffer_size)
        
        # Line history for outlier detection
        self.left_positions = deque(maxlen=10)
        self.right_positions = deque(maxlen=10)
        
    def preprocess_frame(self, frame):
        """Convert frame to grayscale, apply Gaussian blur and color filtering"""
        # Make a copy of the frame
        self.original_frame = frame.copy()
        
        # Get frame dimensions
        self.height, self.width = frame.shape[:2]
        
        # Default ROI if not specified - make it more focused on the road
        if self.roi_vertices is None:
            # More conservative region to avoid detecting irrelevant objects
            self.roi_vertices = np.array([
                [(self.width//5, self.height), 
                 (self.width//2 - 60, self.height//2 + 50), 
                 (self.width//2 + 60, self.height//2 + 50), 
                 (self.width*4//5, self.height)]
            ])
        
        # Set default lane width if not specified
        if self.lane_width_px is None:
            self.lane_width_px = self.width // 3
        
        # Apply color filtering to isolate white and yellow lane markings
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White color mask
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow color mask
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply mask to original frame
        filtered_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur with larger kernel for more smoothing
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Store for visualization
        self.color_filtered = filtered_frame
        
        return blurred
    
    def detect_edges(self, frame):
        """Apply Canny edge detection"""
        return cv2.Canny(frame, self.canny_low_threshold, self.canny_high_threshold)
    
    def region_of_interest(self, edges):
        """Apply a mask to focus only on the region of interest"""
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # For visualization
        self.roi_mask = mask
        
        return masked_edges
    
    def detect_lines(self, edges):
        """Use Hough transform to detect lines"""
        return cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
    
    def separate_lines(self, lines):
        """Separate detected lines into left and right lane lines with improved filtering"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
            
        # Define the expected left and right lane positions
        # (for filtering lines that are too far from expected position)
        mid_x = self.width // 2
        left_region_x = mid_x - self.lane_width_px
        right_region_x = mid_x + self.lane_width_px
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip if points are the same
            if x1 == x2 and y1 == y2:
                continue
                
            # Calculate slope
            if x2 - x1 == 0:
                continue  # Skip vertical lines
                
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate the y-intersection with the bottom of the frame
            # y = mx + b => b = y - mx
            b = y1 - slope * x1
            # Calculate x at y = height (bottom of frame)
            x_at_bottom = int((self.height - b) / slope) if slope != 0 else x1
            
            # Filter based on slope and positioning
            # For left lane lines: negative slope within reasonable range and position
            # For right lane lines: positive slope within reasonable range and position
            if (-self.max_slope <= slope <= -self.min_slope and 
                0 < x_at_bottom < mid_x + 20):  # Allow small margin for crossing center
                left_lines.append(line)
            elif (self.min_slope <= slope <= self.max_slope and 
                  mid_x - 20 < x_at_bottom < self.width):  # Allow small margin
                right_lines.append(line)
                
        return left_lines, right_lines
    
    def average_lines(self, lines, side):
        """Calculate average line parameters and apply outlier filtering"""
        if not lines:
            # Return values from buffer if available
            if side == 'left' and self.left_lines_buffer:
                return self.left_lines_buffer[-1]
            elif side == 'right' and self.right_lines_buffer:
                return self.right_lines_buffer[-1]
            return None
            
        # Extract points
        slopes = []
        lengths = []
        x_at_bottoms = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope and length
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate x at bottom of frame
            b = y1 - slope * x1
            x_at_bottom = int((self.height - b) / slope) if slope != 0 else x1
            
            slopes.append(slope)
            lengths.append(length)
            x_at_bottoms.append(x_at_bottom)
        
        # Get median values as they are more robust to outliers
        median_slope = np.median(slopes)
        median_x = np.median(x_at_bottoms)
        
        # Check if current value is too far from history
        if side == 'left' and self.left_positions:
            avg_hist_x = np.mean(self.left_positions)
            # If new median is far from historical average, use historical data instead
            if abs(median_x - avg_hist_x) > 50:  # Threshold for outlier detection
                if self.left_lines_buffer:
                    return self.left_lines_buffer[-1]
        elif side == 'right' and self.right_positions:
            avg_hist_x = np.mean(self.right_positions)
            if abs(median_x - avg_hist_x) > 50:
                if self.right_lines_buffer:
                    return self.right_lines_buffer[-1]
        
        # Calculate line endpoints using the median values
        y1 = self.height  # Bottom of the frame
        y2 = int(self.height * 0.6)  # Somewhat above the middle
        
        x1 = int(median_x)  # X at bottom of frame
        x2 = int(x1 + (y2 - y1) / median_slope) if median_slope != 0 else x1
        
        result = np.array([[x1, y1, x2, y2]])
        
        # Update position history
        if side == 'left':
            self.left_positions.append(x1)
            self.left_lines_buffer.append(result)
        else:
            self.right_positions.append(x1)
            self.right_lines_buffer.append(result)
            
        return result
    
    def draw_lines(self, frame, left_line, right_line):
        """Draw lane lines on the frame"""
        line_image = np.zeros_like(frame)
        
        # Draw left lane line
        if left_line is not None:
            x1, y1, x2, y2 = left_line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red
        
        # Draw right lane line
        if right_line is not None:
            x1, y1, x2, y2 = right_line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red
        
        # Fill lane area
        if left_line is not None and right_line is not None:
            left_x1, left_y1, left_x2, left_y2 = left_line[0]
            right_x1, right_y1, right_x2, right_y2 = right_line[0]
            
            # Create a polygon and fill it
            vertices = np.array([
                [(left_x1, left_y1), 
                 (left_x2, left_y2), 
                 (right_x2, right_y2), 
                 (right_x1, right_y1)]
            ])
            cv2.fillPoly(line_image, vertices, (0, 255, 0, 0.3))  # Green with transparency
        
        # Combine with original frame
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        # Draw ROI outline for debugging
        roi_debug = frame.copy()
        cv2.polylines(roi_debug, self.roi_vertices, True, (0, 255, 255), 2)
        
        # Store ROI overlay for visualization
        self.roi_overlay = roi_debug
        
        return result
    
    def process_frame(self, frame):
        """Process a frame through the lane detection pipeline"""
        # Preprocess with color filtering
        processed = self.preprocess_frame(frame)
        
        # Edge detection
        edges = self.detect_edges(processed)
        
        # Region of interest
        roi = self.region_of_interest(edges)
        
        # Line detection
        lines = self.detect_lines(roi)
        
        # Separate lines with improved filtering
        left_lines, right_lines = self.separate_lines(lines)
        
        # Average and smooth lines with outlier detection
        left_line = self.average_lines(left_lines, 'left')
        right_line = self.average_lines(right_lines, 'right')
        
        # Draw lines
        result = self.draw_lines(frame, left_line, right_line)
        
        return result
    
    def visualize_steps(self, frame):
        """Visualize all the steps in the pipeline for debugging"""
        # Process the frame
        processed = self.preprocess_frame(frame)
        edges = self.detect_edges(processed)
        roi = self.region_of_interest(edges)
        result = self.process_frame(frame)
        
        # Create a visualization grid
        h, w = frame.shape[:2]
        viz = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Convert single channel images to 3 channels for visualization
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        # Place images in grid
        viz[0:h, 0:w] = frame  # Original
        viz[0:h, w:w*2] = self.color_filtered  # Color filtered
        viz[h:h*2, 0:w] = edges_color  # Edges with ROI
        viz[h:h*2, w:w*2] = result  # Final result with lane overlay
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(viz, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, "Color Filtered", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, "Edge Detection + ROI", (10, h + 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, "Lane Detection Result", (w + 10, h + 30), font, 0.7, (255, 255, 255), 2)
        
        # Draw ROI on the original image for reference
        cv2.polylines(viz[0:h, 0:w], self.roi_vertices, True, (0, 255, 255), 2)
        
        return viz


def main():
    """Main function to run the improved lane detection"""
    # Parameters (these can be adjusted for different conditions)
    params = {
        'canny_low_threshold': 70,
        'canny_high_threshold': 170,
        'hough_threshold': 40,
        'min_line_length': 60,
        'max_line_gap': 30,
        'smoothing_buffer_size': 15,
        'min_slope': 0.5,
        'max_slope': 2.0
    }
    
    # Initialize detector
    detector = ImprovedLaneDetector(**params)
    
    # Check if video file exists
    video_path = "road_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Please place a road video file named 'road_video.mp4' in the same directory")
        print("or update the video_path variable with the correct path to your video file.")
        return

    # Load video
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('lane_detection_improved_output.avi', fourcc, fps, (width, height))
    
    # Set up visualization video writer (optional)
    viz_out = cv2.VideoWriter('lane_detection_visualization.avi', fourcc, fps, (width, height))
    
    # Process video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        # Process frame
        result = detector.process_frame(frame)
        
        # Write to output video
        out.write(result)
        
        # Create visualization (optional)
        viz = detector.visualize_steps(frame)
        viz_resized = cv2.resize(viz, (width, height))
        viz_out.write(viz_resized)
        
        # Show result
        cv2.imshow('Lane Detection', result)
        cv2.imshow('Lane Detection Steps', cv2.resize(viz, (width, height)))
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    video.release()
    out.release()
    viz_out.release()
    cv2.destroyAllWindows()
    
    print("Processing complete!")
    print("Output videos saved as 'lane_detection_improved_output.avi' and 'lane_detection_visualization.avi'")


if __name__ == "__main__":
    main() 