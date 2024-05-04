import cv2

frames=[]
N = 2
THRESH = 60
ASSIGN_VALUE = 255 #Value to assign the pixel if the threshold is met

cap = cv2.VideoCapture(0)  #Capture using Computer's Webcam
old_frame = None
    
while True:
    #Capture frame by frame
    ret, frame = cap.read()
    #Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    #Append to list of frames
    #D(N) = || I(t) - I(t+N) || = || I(t-N) - I(t) ||
    if old_frame is not None:
        diff = cv2.absdiff(old_frame, frame_gray)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)
    #Mask Thresholding
        # threshold_method = cv2.THRESH_BINARY
        edges = cv2.Canny(diff, 35, THRESH)
        # ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, threshold_method)
        #Display the Motion Mask
        cv2.imshow('Motion Mask', edges)
        cv2.imshow('Difference', diff)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Original', frame)
    #Wait for 1ms for a key to be pressed
    old_frame = frame_gray
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()