import numpy as np
import cv2
import face_recognition

imgElon = face_recognition.load_image_file('Images\Elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgBill = face_recognition.load_image_file('Images\BillG.jpg')
imgBill = cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocB = face_recognition.face_locations(imgBill)[0]
encodeBillG = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeBillG)
facedis = face_recognition.face_distance([encodeElon],encodeBillG)
print(results,facedis)
cv2.imshow('Elon',imgElon)
cv2.imshow('Bill',imgBill)
cv2.waitKey(0)