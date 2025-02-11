"""
--------------------------
Real-Time Document Scanner
--------------------------
An OpenCV-based document scanner that detects documents in real-time.
- Auto-detects and crops documents using OpenCV.
- Displays "Edged" image in real-time for detection feedback.
- Press 'G' to manually select four corners from the last real-time frame.
- Press 'B' to save the latest auto-detected document as a static image.
- Press 'ESC' to exit.
"""

import cv2
import numpy as np

# Global variables
manual_points = []
image_for_selection = None
manual_mode = False  
saved_image = None  
latest_scanned_document = None  
last_real_time_frame = None  

# Function to order points for perspective transformation
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

# Function to apply four-point perspective transformation
def four_point_perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# Mouse callback function for manual point selection
def select_points(event, x, y, flags, param):
    global manual_points, image_for_selection
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 4:
            manual_points.append((x, y))
            cv2.circle(image_for_selection, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 4 Points", image_for_selection)

# Function to detect document in a frame
def detect_document(frame):
    height = 800
    ratio = frame.shape[0] / float(height)
    resized_width = int(frame.shape[1] / ratio)
    resized_image = cv2.resize(frame, (resized_width, height))

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx.reshape(4, 2) * ratio
            return screenCnt.astype(int), edges  

    return None, edges  

# Initialize webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    last_real_time_frame = frame.copy()  

    # Detect document
    screenCnt, edged_image = detect_document(frame)

    if screenCnt is not None:
        latest_scanned_document = four_point_perspective_transform(frame, screenCnt)
        cv2.imshow("Scanned Document", latest_scanned_document)  
        
    # Draw instruction text
    cv2.putText(frame, "Press G to Manual Crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green
    cv2.putText(frame, "Press B to Select Auto", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Blue

    # Show frames
    cv2.imshow("Real-Time Scanner", frame)
    cv2.imshow("Edged", edged_image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):  
        if latest_scanned_document is not None:
            saved_image = latest_scanned_document.copy()
            cv2.imshow("Saved Document", saved_image)  

    if key == ord('g'):  
        manual_mode = True
        image_for_selection = last_real_time_frame.copy()  
        manual_points = []

        # Show the real-time frame for manual selection
        cv2.imshow("Select 4 Points", image_for_selection)
        cv2.setMouseCallback("Select 4 Points", select_points)

        # Wait for user to select 4 points
        while True:
            cv2.waitKey(1)
            if len(manual_points) == 4:
                break
        
        screenCnt = np.array(manual_points, dtype="float32")
        latest_scanned_document = four_point_perspective_transform(image_for_selection, screenCnt)

        # Display the manually cropped document as a static image
        cv2.imshow("Manually Scanned Document", latest_scanned_document)

        manual_mode = False  

    elif key == 27:  
        break

cap.release()
cv2.destroyAllWindows()
