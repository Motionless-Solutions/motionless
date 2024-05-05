import cv2

LEARNING_RATE = 0.5  # Learning rate for the background model
KERNEL_SIZE = 5  # Kernel size for morphological operations
N_OPEN = 15  # Number of iterations for morphological opening operation

fgbg = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Apply MOG
    motion_mask = fgbg.apply(frame, LEARNING_RATE)  # type: ignore
    # Get background
    background = fgbg.getBackgroundImage()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel, iterations=N_OPEN)
    # Display the motion mask and background
    cv2.imshow("background", background)
    cv2.imshow("Motion Mask", motion_mask)
    cv2.imshow("Original", frame)
    # Wait for 1ms for a key to be pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
