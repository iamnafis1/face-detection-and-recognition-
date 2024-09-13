import cv2
import numpy as np
import face_recognition as fr


img=cv2.imread("Nafish.jpg")
rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_encoding=fr.face_encodings(rgb_img)[0]

img2=cv2.imread("images/Nafish.jpeg")
rgb_img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img_encoding2=fr.face_encodings(rgb_img2)[0]



result=fr.compare_faces([img_encoding],img_encoding2)
print("Result: ",result)

cv2.imshow("img",img)
cv2.imshow("img2",img2)
cv2.waitKey(0)

# image = fr.load_image_file("Nafish.jpg")
# face_locations = fr.face_locations(image)