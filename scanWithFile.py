\
# Import necessary packages
import cv2
import os
import numpy as np

# Global variables for manual selection
manual_points = []
image_for_selection = None

# Function to order points for perspective transformation
def order_points(pts):
    """ Orders the points in the order: top-left, top-right, bottom-right, bottom-left """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference of points to determine order
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

# Function to apply four-point perspective transformation
def four_point_perspective_transform(image, pts):
    """ Performs a perspective transformation using the given four points """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for warp
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and apply warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Mouse click callback function for manual point selection
def select_points(event, x, y, flags, param):
    global manual_points, resized_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 4:
            manual_points.append((x * ratio, y * ratio))  # Scale back to original image
            cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 4 Points", resized_image)

# Load the image
image_PATH = r"Document-Scanner-master\images\scrapIMg.jpg"
if not os.path.exists(image_PATH):
    print(f"Error: Image file '{image_PATH}' not found.")
    exit(1)

image = cv2.imread(image_PATH)
if image is None:
    print(f"Error: Unable to read image '{image_PATH}'. Check the file path or file format.")
    exit(1)

# Make copies of the original image
orig = image.copy()

# Resize for processing
height = 800  # Fixed processing height
ratio = image.shape[0] / float(height)
resized_width = int(image.shape[1] / ratio)
resized_image = cv2.resize(image, (resized_width, height))

# Convert to grayscale and detect edges
image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
image_edge = cv2.Canny(image_blurred, 75, 200)

cv2.imshow("Image", resized_image)
cv2.imshow("Edged", image_edge)

# Find contours
cnts = cv2.findContours(image_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# Try to find a 4-point contour automatically
screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

# If no automatic detection, enter manual mode
if screenCnt is None:
    print("Automatic document detection failed. Please select 4 points manually.")

    cv2.imshow("Select 4 Points", resized_image)
    cv2.setMouseCallback("Select 4 Points", select_points)

    # Wait until 4 points are selected
    while True:
        cv2.waitKey(1)
        if len(manual_points) == 4:
            break

    # Convert selected points to NumPy array (already scaled back to original)
    screenCnt = np.array(manual_points, dtype="float32")

else:
    # Scale the contour points back to original image size
    screenCnt = screenCnt.reshape(4, 2) * ratio

# Ensure correct point order for perspective transform
screenCnt = order_points(screenCnt)

# Apply perspective transformation
warped_image = four_point_perspective_transform(orig, screenCnt.astype(int))

# Check if the transformation was successful
if warped_image is None or warped_image.size == 0:
    print("Error: The warped image is empty. Check the perspective transform function.")
else:
    # Display the scanned image
    cv2.imshow("Scanned Document", warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
