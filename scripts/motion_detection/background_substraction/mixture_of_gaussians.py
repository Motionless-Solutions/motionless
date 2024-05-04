import cv2

MAX_FRAMES = 1000
LEARNING_RATE = -1   
fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #Apply MOG
    motion_mask = fgbg.apply(frame, LEARNING_RATE)
    #Get background
    background = fgbg.getBackgroundImage()
    # Display the motion mask and background
    cv2.imshow('background', background)
    cv2.imshow('Motion Mask', motion_mask)
    cv2.imshow('Original', frame)
    # Wait for 1ms for a key to be pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()