import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

class LaneDetector:
    def __init__(self, 
                 canny_low_threshold=50, 
                 canny_high_threshold=150,
                 hough_threshold=30,
                 min_line_length=40,
                 max_line_gap=20,
                 roi_vertices=None,
                 smoothing_buffer_size=10):
        """
        Initialize the lane detector with configurable parameters
        
        Parameters:
        -----------
        canny_low_threshold : int
            Lower threshold for Canny edge detector
        canny_high_threshold : int
            Upper threshold for Canny edge detector
        hough_threshold : int
            Accumulator threshold for Hough transform
        min_line_length : int
            Minimum line length for Hough lines
        max_line_gap : int
            Maximum allowed gap between line segments
        roi_vertices : numpy array
            Custom region of interest vertices
        smoothing_buffer_size : int
            Number of frames to use for smoothing lane lines
        """
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.roi_vertices = roi_vertices
        
        # Buffers for smoothing
        self.left_lines_buffer = deque(maxlen=smoothing_buffer_size)
        self.right_lines_buffer = deque(maxlen=smoothing_buffer_size)
        
    def preprocess_frame(self, frame):
        """Convert frame to grayscale and apply Gaussian blur"""
        # Make a copy of the frame
        self.original_frame = frame.copy()
        
        # Get frame dimensions
        self.height, self.width = frame.shape[:2]
        
        # Default ROI if not specified
        if self.roi_vertices is None:
            self.roi_vertices = np.array([
                [(50, self.height), 
                 (self.width//2 - 45, self.height//2 + 60), 
                 (self.width//2 + 45, self.height//2 + 60), 
                 (self.width - 50, self.height)]
            ])
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
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
        """Separate detected lines into left and right lane lines"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:
                continue  # Skip vertical lines
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter based on slope
            if abs(slope) < 0.5:  # Skip horizontal-ish lines
                continue
                
            if slope < 0:  # Negative slope = left lane
                left_lines.append(line)
            else:  # Positive slope = right lane
                right_lines.append(line)
                
        return left_lines, right_lines
    
    def average_lines(self, lines, side):
        """Calculate average line parameters and return line points"""
        if not lines:
            # Return values from buffer if available
            if side == 'left' and self.left_lines_buffer:
                return self.left_lines_buffer[-1]
            elif side == 'right' and self.right_lines_buffer:
                return self.right_lines_buffer[-1]
            return None
        
        # Extract points and calculate slopes and intercepts
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Use polyfit to get line parameters
        if len(x_coords) >= 2:  # Need at least 2 points for a line
            parameters = np.polyfit(x_coords, y_coords, 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            # Calculate line endpoints
            y1 = self.height  # Bottom of the frame
            y2 = int(self.height * 0.6)  # Somewhat above the middle
            
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            
            result = np.array([[x1, y1, x2, y2]])
            
            # Store in buffer for smoothing
            if side == 'left':
                self.left_lines_buffer.append(result)
            else:
                self.right_lines_buffer.append(result)
                
            return result
        
        return None
    
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
        return result
    
    def process_frame(self, frame):
        """Process a frame through the lane detection pipeline"""
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Edge detection
        edges = self.detect_edges(processed)
        
        # Region of interest
        roi = self.region_of_interest(edges)
        
        # Line detection
        lines = self.detect_lines(roi)
        
        # Separate lines
        left_lines, right_lines = self.separate_lines(lines)
        
        # Average and smooth lines
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
        mask_color = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2BGR)
        
        # Place images in grid
        viz[0:h, 0:w] = frame  # Original
        viz[0:h, w:w*2] = edges_color  # Edges
        viz[h:h*2, 0:w] = roi_color  # ROI
        viz[h:h*2, w:w*2] = result  # Final result
        
        # Resize for display
        viz_resized = cv2.resize(viz, (w, h))
        
        return viz_resized


def main():
    """Main function to run the lane detection"""
    # Parameters (these can be adjusted for different conditions)
    params = {
        'canny_low_threshold': 50,
        'canny_high_threshold': 150,
        'hough_threshold': 30,
        'min_line_length': 40,
        'max_line_gap': 20,
        'smoothing_buffer_size': 10
    }
    
    # Initialize detector
    detector = LaneDetector(**params)
    
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
    out = cv2.VideoWriter('lane_detection_output_advanced.avi', fourcc, fps, (width, height))
    
    # Process video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        # Process frame
        result = detector.process_frame(frame)
        
        # Write to output video
        out.write(result)
        
        # Show result
        cv2.imshow('Lane Detection', result)
        
        # Visualization (optional, for debugging)
        # viz = detector.visualize_steps(frame)
        # cv2.imshow('Lane Detection Steps', viz)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 