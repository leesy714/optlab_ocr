import numpy as np
import cv2

img = np.zeros((1024, 1024, 3), np.uint8)
num = 0
for i in range(0, 1024-32, 32):
    for j in range(0, 1024-32, 32):
        img = cv2.rectangle(img, (i, j), (i+32, j+32),
                             (i*j % 255, i*i*j%255, i*j*j%255), 3)
        num += 1
        #cv2.putText(img, str(num), (i, j), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)

cv2.imwrite('image.png', img)
