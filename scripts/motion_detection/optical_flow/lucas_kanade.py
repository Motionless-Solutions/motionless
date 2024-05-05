import cv2
import numpy as np


RANDOM_STATE = 42
cap = cv2.VideoCapture(0)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Create random colors
color = np.random.Generator(bit_generator=np.random.PCG64(seed=RANDOM_STATE)).integers(
    0, 255, (100, 3)
)

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(
    old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    # Read new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,  # type: ignore
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )  # type: ignore
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        pt1 = (int(a), int(b))
        pt2 = (int(c), int(d))
        mask = cv2.line(mask, pt1, pt2, color[i].tolist(), 2)
        frame = cv2.circle(frame, pt1, 5, color[i].tolist(), -1)

    # Display the demo
    img = cv2.add(frame, mask)
    cv2.imshow("frame", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()