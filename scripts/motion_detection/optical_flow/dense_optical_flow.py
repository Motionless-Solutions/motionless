import cv2
import numpy as np

algorithms = [
    "lucaskanade_dense",
    "farneback",
    "rlof"
]
params = []
algorithm = algorithms[0]

match algorithm:
    case "lucaskanade_dense":
        method = cv2.optflow.calcOpticalFlowSparseToDense
        to_grayscale = True
    case "farneback":
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]
        to_grayscale = True
    case "rlof":
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        to_grayscale = False


cap = cv2.VideoCapture(0)
ret, old_frame = cap.read()

# crate HSV & make Value a constant
hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255
if to_grayscale:
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
while True:
    # Read the next frame
    ret, new_frame = cap.read()
    frame_copy = new_frame
    if not ret:
        break

    if to_grayscale:
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = method(
        old_frame,
        new_frame,
        None, # type: ignore
        *params
    ) # type: ignore

    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Use Hue and Value to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2 # type: ignore
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # type: ignore

    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("frame", frame_copy)
    cv2.imshow("optical flow", bgr)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

    # Update the previous frame
    old_frame = new_frame
