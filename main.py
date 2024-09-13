import cv2
from face_rec import SimpleFacerec



#encode faces from a folder
sfr=SimpleFacerec()
sfr.load_encoding_images("images/")

#load Camera
# Get a reference to webcam #0 (the default one)
cap=cv2.VideoCapture(0)



while True:
   # Grab a single frame of video
    ret,frame=cap.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    

    #Detec faces
    face_locations,face_names= sfr.detect_known_faces(frame)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
            
        #     Draw a box around the face
            cv2.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

       # Display the resulting image

    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    if key==27: #esc key on keyBoard
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()