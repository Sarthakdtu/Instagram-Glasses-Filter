
from my_CNN_model import *
import cv2
import numpy as np


my_model = load_my_CNN_model('my_model')

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

kernel = np.ones((5, 5), np.uint8)

filters = ['images/sunglasses.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png', 'images/image1.jpg', 'images/image2.jpeg']
filterIndex = 0

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)
    cv2.putText(frame, "NEXT FILTER", (512, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

   
    (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

   
    if len(cnts) > 0:
   
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
   
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
   
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
   
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 500 <= center[0] <= 620:
                filterIndex += 1
                filterIndex %= 6
                continue

    for (x, y, w, h) in faces:

       
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]
       
        gray_normalized = gray_face / 255
        original_shape = gray_face.shape 
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)

       
        keypoints = my_model.predict(face_resized)

        
        keypoints = keypoints * 48 + 48

       
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

       
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        
        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
        sunglass_width = int((points[7][0]-points[9][0])*1.1)
        sunglass_height = int((points[10][1]-points[8][1])/1.1)
        sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
        transparent_region = sunglass_resized[:,:,:3] != 0
        face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

        
        frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

       
        for keypoint in points:
            cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

        frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)

       
        cv2.imshow("Selfie Filters", frame)
        cv2.imshow("Facial Keypoints", frame2)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
