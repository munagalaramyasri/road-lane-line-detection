import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def convert_hsl(image):
    if len(image.shape) == 2 or image.shape[2] == 1:  # Image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def HSL_color_selection(image):
    converted_image = convert_hsl(image)
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def gray_scale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny_edge_detection(image):
    # Convert image to uint8 if it is not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return cv2.Canny(image, 50, 150)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, lines, color=[255, 0, 0], thickness=12):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    return line_img

def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def process_image(image):
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_blur(gray)
    edges = canny_edge_detection(smooth)
    vertices = np.array([[(0, image.shape[0]), 
                          (image.shape[1] / 2, image.shape[0] / 2), 
                          (image.shape[1], image.shape[0])]], np.int32)
    masked_image = region_of_interest(edges, vertices)
    lines = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 20, np.array([]), minLineLength=20, maxLineGap=300)
    left_line, right_line = lane_lines(image, lines)
    return draw_lane_lines(image, (left_line, right_line))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processed_frame = process_image(frame)
            cv2.imshow('Lane Detection Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# Process all images in the 'images' folder
image_folder = 'image'
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png','.jpeg')):
        image_path = os.path.join(image_folder, filename)
        image = mpimg.imread(image_path)
        processed_image = process_image(image)
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(0)  # Wait for any key press before closing the window
        cv2.destroyAllWindows()

# Process a sample video
video_folder = 'video'  
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
# Process each video file in the list
for video_file in video_files:
    process_video(video_file)
