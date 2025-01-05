import datetime
import os
import threading
from tkinter import simpledialog
from turtle import pd
import cv2
from mtcnn import MTCNN


def TrackImages():
    subject_name = simpledialog.askstring("Input", "Enter Subject Name:")
    if subject_name:
        # Initialize the recognizer and load the trained model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        
        # Initialize MTCNN for face detection
        detector = MTCNN()
        
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        cam = cv2.VideoCapture(0)
        
        col_names = ['Id', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)
        recognized_ids = set()  # To track recognized IDs

        while True:
            ret, im = cam.read()
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert to RGB for MTCNN
            faces = detector.detect_faces(rgb_im)  # Detect faces using MTCNN

            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']

                if confidence > 0.95:  # Confidence threshold
                    face_region = im[y:y+h, x:x+w]  # Extract face region

                    # Recognize the face using the trained recognizer
                    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    Id, conf = recognizer.predict(gray_face)

                    if conf < 70:  # Confidence threshold for recognition
                        ts = datetime.time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
                        aa = df.loc[df['Id'] == Id]['Name'].values
                        name = aa[0] if len(aa) > 0 else "Unknown"

                        # Add to attendance only if not already present
                        if Id not in recognized_ids:
                            attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                            recognized_ids.add(Id)

                            # Real-time update in terminal
                            print(f"Attendance marked for {name} (ID: {Id})")

                        # Draw rectangle around the face and display ID and Name
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
                        text = f'ID: {Id} Name: {name}'
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(im, (x, y - 20), (x + text_size[0], y), (0, 255, 0), -1)  # Green filled rectangle
                        cv2.putText(im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    else:
                        Id = 'Unknown'
                        # Optionally handle unknown cases (e.g., draw a red rectangle)

            cv2.imshow('Facial Recognition', im)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save attendance to CSV in the respective subject folder
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.now().strftime('%H-%M-%S')
        folder_path = f"Attendance/{subject_name}"
        os.makedirs(folder_path, exist_ok=True)
        attendance_file = f"{folder_path}/Attendance_{date}_{timeStamp}.csv"
        attendance.to_csv(attendance_file, index=False)

        acknowledgment_label.configure(text="Attendance Updated")
        cam.release()
        threading.Thread(target=play_attendance_updated_sound).start()
        cv2.destroyAllWindows()
