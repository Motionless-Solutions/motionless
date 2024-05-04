import cv2
import numpy as np

background = None
THRESH = 50
ASSIGN_VALUE = 255
ALPHA = 0.1

def update_background(current_frame, prev_bg, alpha) -> np.ndarray:
    bg = alpha * current_frame + (1 - alpha) * prev_bg
    bg = np.array(bg, dtype=np.uint8)
    return bg

cap = cv2.VideoCapture(0)
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)      
    if background is None:
        # Train background with first frame
        background = frame_gray
    else:
        # Background subtraction
        diff = cv2.absdiff(background, frame_gray)
        background = update_background(frame_gray, background, alpha=ALPHA)
        # Mask thresholding
        # diff = cv2.divide(diff, background)
        # diff = cv2.multiply(diff, np.ones_like(diff, dtype=np.uint8), scale=255)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)
        ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, cv2.THRESH_BINARY)
        # dilute the mask to fill in holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.erode(motion_mask, kernel, iterations=3)
        # motion_mask = cv2.dilate(motion_mask, kernel, iterations=3)
        # Find contour
        contours, hierarchy = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw rectangle around outer contours
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # Display the motion mask and background
        cv2.imshow('Motion mask', motion_mask)
        cv2.imshow('Background', background)

    cv2.imshow('Original', frame)
    # Wait for 1ms for a key to be pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()