import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3 as textspeech


engine = textspeech.init()

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


studentimg = []
studentName = []
myList = os.listdir()
#print(myList)
# filrtered list to include only image files
image_extensions = ['.jpg', '.jpeg', '.png']
valid_image_files = [i for i in myList if os.path.splitext(i)[1].lower() in image_extensions]

for i in valid_image_files:
    studentimg.append(i)
    studentName.append(os.path.splitext(i)[0])
# print(studentName)


def finEncoding(images):
    imgEncodings = []
    for img_filename in images:
        img = cv2.imread(img_filename)
        img = resize(img, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodimg = face_recognition.face_encodings(img)[0]
        imgEncodings.append(encodimg)
    return imgEncodings



def MarkAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            time_str = now.strftime('%H: %M: %S')
            f.writelines(f'\n{name}, {time_str}')
            engine.say('hi' + name + 'hope you are doing good')
            engine.runAndWait()




EncodeList = finEncoding(valid_image_files)

capture = cv2.VideoCapture(0)
while True:
    success, frame = capture.read()
    smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    facesInFrame = face_recognition.face_locations(smaller_frames)
    encodeFacesInFrame = face_recognition.face_encodings(smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        matches = face_recognition.compare_faces(EncodeList, encodeFace)
        facedis = face_recognition.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = studentName[matchindex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1-6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)
    cv2.imshow('capture', frame)
    cv2.waitKey(1)

