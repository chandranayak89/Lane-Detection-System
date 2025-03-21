# Lane Detection System

This repository contains a beginner-friendly implementation of a lane detection system using computer vision techniques. This project can be used as a starting point for autonomous vehicle development or advanced driver-assistance systems (ADAS).

## Project Overview

Lane detection is a fundamental component of autonomous driving systems. This project demonstrates how to:

1. Process video frames to identify lane markings
2. Apply computer vision techniques like edge detection and Hough transforms
3. Extract and visualize lane boundaries in real-time

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib

You can install the required dependencies using:

```bash
pip install opencv-python numpy matplotlib
```

## Project Structure

- `lane_detection.py` - Basic lane detection implementation
- `advanced_lane_detection.py` - Enhanced implementation with additional features
- `improved_lane_detection.py` - Further improved implementation with better filtering for reducing false positives
- `lane_detector_tuning.py` - Interactive tool for parameter tuning
- `README.md` - Project documentation (this file)

## Usage

### Basic Lane Detection

The basic version implements the fundamental pipeline for lane detection:

```bash
python lane_detection.py
```

### Advanced Lane Detection

The advanced version includes:
- Improved lane line detection and filtering
- Line smoothing across frames
- Lane area highlighting
- Debug visualization options
- Configurable parameters

```bash
python advanced_lane_detection.py
```

### Improved Lane Detection (Reduced False Positives)

The improved version addresses common issues with false positives by adding:
- Color filtering to isolate white and yellow lane markings
- More robust line filtering based on slope and position
- Outlier detection and rejection
- Temporal smoothing with larger buffer for stability
- More conservative ROI for excluding irrelevant objects

```bash
python improved_lane_detection.py
```

### Interactive Parameter Tuning

This tool allows you to adjust detection parameters in real-time to optimize for your specific video:

```bash
python lane_detector_tuning.py
```

The tuning interface provides sliders for adjusting:
- Canny edge detection thresholds
- Hough transform parameters
- Slope thresholds for line filtering
- ROI (Region of Interest) dimensions

Press 'p' during tuning to print the current parameter values, which you can then use in your code.

## How It Works

### Lane Detection Pipeline

1. **Image Preprocessing**:
   - Convert to grayscale to simplify processing
   - Apply Gaussian blur to reduce noise
   - **New**: Apply color filtering to isolate white and yellow lane markings

2. **Edge Detection**:
   - Use Canny edge detection to identify strong gradients in the image

3. **Region of Interest Selection**:
   - Apply a mask to focus only on the road area where lanes are expected
   - **New**: More conservative ROI to avoid detecting objects outside the lane area

4. **Line Detection**:
   - Use Hough Transform to detect straight lines in the edge image

5. **Line Filtering and Averaging**:
   - Separate detected lines into left and right lane markings
   - **New**: Filter lines by slope and position to remove false positives
   - **New**: Use median values for more robustness against outliers
   - Average and extrapolate lane lines
   - Apply temporal smoothing across frames
   - **New**: Outlier detection to maintain lane stability

6. **Visualization**:
   - Draw the detected lane lines on the original frame
   - Highlight the lane area between detected lines
   - **New**: Debug visualization mode showing each step of the process

## Customization

The improved implementation allows for customization of various parameters:

- Edge detection thresholds
- Hough transform parameters
- Region of interest vertices
- Line smoothing buffer size
- Slope thresholds for filtering
- Color filtering ranges

For best results:
1. Run the parameter tuning script (`lane_detector_tuning.py`)
2. Adjust parameters to get optimal lane detection for your video
3. Copy the printed parameters to your own implementation

## Troubleshooting False Positives

If you're experiencing false positive lane detections:

1. **Adjust ROI**: Make your region of interest more focused on the actual road area
2. **Increase Canny thresholds**: Higher thresholds reduce noise but may miss faint lane markings
3. **Increase Hough threshold**: Requires more points to form a line, reducing spurious detections
4. **Adjust slope thresholds**: Filter out lines with unrealistic slopes
5. **Increase min line length**: Ignore short line segments that might be noise
6. **Use color filtering**: Isolate only white and yellow colors typical of lane markings

## Sample Results

When running the code with a suitable road video, you should see:
- Red lines indicating the detected lane boundaries
- Green shaded area representing the detected lane

## Limitations and Future Improvements

This implementation has several limitations:

1. **Lighting conditions**: Performance may degrade in poor lighting, shadows, or glare
2. **Curved roads**: The basic implementation assumes straight lane lines
3. **Occlusions**: Lane markings blocked by other vehicles may cause detection issues

Potential improvements:
- Implement curved lane detection using polynomial fitting
- Add more sophisticated color filtering to better isolate yellow and white lane markings
- Apply deep learning approaches for more robust detection
- Implement Kalman filtering for smoother tracking

## Contributing

Contributions are welcome! Feel free to improve the code, add features, or fix issues.

## License

This project is released under the MIT License.

## Acknowledgments

This project is intended as an educational resource for computer vision and autonomous vehicle development.

## Contact

If you have any questions or suggestions, please open an issue in this repository. 