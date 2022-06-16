
import cv2
import numpy as np
import face_recognition as fr
import os


from datetime import datetime





path = "./Image_check"

imgs = []

classNames = []
myList = os.listdir(path)
print(myList)


for cl in myList:
    curI = cv2.imread(f'{path}/{cl}')
    imgs.append(curI)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)



def find_endcode(imgs):
    encodeList = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList



def markRE(name):
    with open('history.csv', 'rt') as f:
        myData = f.readlines()
        nameList = []

        for line in myData:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')








encodeListKnown = find_endcode(imgs)
# print(len(encodeListKnown))
print("Encoding Complete")


cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame= fr.face_locations(imgS)
    encodeCurFrame = fr.face_encodings(imgS, faceCurFrame)


    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2,y2,x1 = faceLoc
            # y1, x2,y2,x1 = y1*4, x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2)
            # markRE(name)


     
    cv2.imshow("webcam", img)
    cv2.waitKey(1)





# faceLoc= fr.face_locations(imgDuy)[0]
# encodeDuy = fr.face_encodings(imgDuy)[0]
# cv2.rectangle(imgDuy, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255,0,255),2)


# faceLocTest= fr.face_locations(imgDuyTest)[0]
# encodeDuyTest = fr.face_encodings(imgDuyTest)[0]
# cv2.rectangle(imgDuyTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]), (255,0,255),2)




# result = fr.compare_faces([encodeDuy], encodeDuyTest) 
# faceDis = fr.face_distance([encodeDuy], encodeDuyTest)