import cv2
import numpy as np

THRESH = 50  # Threshold for motion detection, when the difference between the background and the current frame is greater than THRESH, it is considered as motion
ALPHA = 0.1  # Learning rate for background update, when 0.1, the background is updated by 10% of the current frame and 90% of the previous background
GAUSSIAN_BLUR_SIZE = 7  # Gaussian blur kernel size
KERNEL_SIZE = 5  # Kernel size for morphological operations
N_CLOSE = 20  # Number of iterations for morphological closing operation
MIN_AREA = 500  # Minimum area for a contour to be considered as motion


def update_background(current_frame, prev_bg, alpha) -> np.ndarray:
    bg = alpha * current_frame + (1 - alpha) * prev_bg
    bg = np.array(bg, dtype=np.uint8)
    return bg


cap = cv2.VideoCapture(0)
# Capture the first frame to initialize the background
_, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
background = frame_gray

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Background subtraction
    diff = cv2.absdiff(background, frame_gray)
    background = update_background(frame_gray, background, alpha=ALPHA)
    # Mask thresholding
    # diff = cv2.divide(diff, background)
    # diff = cv2.multiply(diff, np.ones_like(diff, dtype=np.uint8), scale=255)
    diff = cv2.GaussianBlur(diff, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    ret, motion_mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=N_CLOSE)
    # Find contour
    contours, hierarchy = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw rectangle around outer contours
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the motion mask and background
    cv2.imshow("Motion Mask", motion_mask)
    cv2.imshow("Background", background)

    cv2.imshow("Original", frame)
    # Wait for 1ms for a key to be pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
