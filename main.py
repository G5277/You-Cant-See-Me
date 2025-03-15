import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.waitKey(2000)
ret, init_frame = cap.read()
if not ret:
    print("Error capturing the bg frame.")
    cap.release()
    exit()

print("hello")

lower_hsv = np.array([90, 100, 50])   # Covers dark blue
upper_hsv = np.array([130, 255, 255]) # Covers bright and light blue

kernel = np.ones((3,3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing the frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 5) 
    mask = cv2.dilate(mask, kernel, iterations=5) 

    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    inv_area = cv2.bitwise_and(init_frame, init_frame, mask=mask)

    final = cv2.addWeighted(bg, 1, inv_area, 1, 0)

    cv2.imshow("You Can't See Me", final)

    if cv2.waitKey(3) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()