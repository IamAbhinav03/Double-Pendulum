import cv2
import imutils

path = "sample.mp4"

cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("cap not opened")

xstart = 750
xend = 1850
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        resized_frame = imutils.resize(frame, width=600)
        print(f"rfs: {resized_frame.shape}")
        resized_frame = resized_frame[:, 200:600]
        print(f"rfs: {resized_frame.shape}")
        cv2.imshow("resized frame", resized_frame)
        # Cropping an image
        print(frame.shape)
        cropped_image = frame[150:, xstart:xend]
 
        # Display cropped image
        cv2.imshow("cropped", cropped_image)
        key = cv2.waitKey(0)
        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("n"):
            # print("input values for xstart")
            # xstart = int(input())
            # xend = int(input("input values for xend"))
            continue
    else:
        print("End of frame")

cv2.destroyAllWindows()
cap.release()