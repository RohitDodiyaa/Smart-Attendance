import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import threading
import pygame
from PIL import Image, ImageEnhance
from mtcnn import MTCNN


# Defining sounds
def play_welcome_sound():
    pygame.mixer.music.load('Welcome-to-Face-Recognize-system.mp3')
    pygame.mixer.music.play()

def play_dataset_creation_sound():
    pygame.mixer.music.load('DatasetCreated.mp3')
    pygame.mixer.music.play()

def play_dataset_trained_sound():
    pygame.mixer.music.load('Dataset-Trained-Successfully.mp3')
    pygame.mixer.music.play()

def play_attendance_updated_sound():
    pygame.mixer.music.load('ThankyouYourattendanceupdated.mp3')
    pygame.mixer.music.play()

# Initialize tkinter window and pygame
window = tk.Tk()
window.title("Attendance MS")
pygame.mixer.init()

# Start playing the welcome sound in a separate thread
threading.Thread(target=play_welcome_sound).start()

# Load and display an image
path = 'face2.jpg'
img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image=img)
panel.image = img
panel.pack(side="bottom", fill="both", expand="yes")

# UI labels
message = tk.Label(window, text="Attendance Tracking System", bg="white", fg="black", width=21, height=1, font=('times', 30, 'bold'))
message.place(x=10, y=10)

# Entry fields for ID and Name
lbl = tk.Label(window, text="Enter ID:", width=10, height=1, fg="black", bg="white", font=('times', 13, ' bold '))
lbl.place(x=55, y=195)

txt = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt.place(x=180, y=195)

lbl2 = tk.Label(window, text="Enter Name:", width=10, fg="black", bg="white", height=1, font=('times', 13, ' bold '))
lbl2.place(x=55, y=226)

txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt2.place(x=180, y=226)

# Acknowledgment label
acknowledgment_label = tk.Label(window, text="Welcome!!", fg="black", bg="white", width=60, height=2, font=('times', 10, 'italic bold'))
acknowledgment_label.place(x=30, y=365)

# Main functionality for taking images
def clear_entries():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def augment_image(image):
    """Apply various augmentations to the given image."""
    augmented_images = []

    # Convert the image to a PIL image for augmentation
    pil_image = Image.fromarray(image)

    # Flip the image horizontally
    flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append(np.array(flipped))

    # Rotate the image at different angles
    for angle in range(-15, 16, 5):  # Rotating from -15 to +15 degrees
        rotated = pil_image.rotate(angle)
        augmented_images.append(np.array(rotated))

    # Apply contrast and brightness enhancements
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(1.5)  # Increase contrast
    augmented_images.append(np.array(enhanced))

    enhancer = ImageEnhance.Brightness(pil_image)
    bright = enhancer.enhance(1.3)  # Increase brightness
    augmented_images.append(np.array(bright))

    return augmented_images

def TakeImagesWithAugmentation():
    Id = txt.get()
    name = txt2.get()

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        detector = MTCNN()  # Use MTCNN for face detection
        sampleNum = 0

        while True:
            ret, img = cam.read()
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for MTCNN
            faces = detector.detect_faces(rgb_img)  # Detect faces using MTCNN

            for face in faces:
                x, y, w, h = face['box']  # Bounding box for the face
                confidence = face['confidence']

                if confidence > 0.95:  # Confidence threshold
                    sampleNum += 1
                    face_img = img[y:y+h, x:x+w]

                    # Save the original face image with JPEG quality setting
                    cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # Apply data augmentation and save augmented images
                    augmented_faces = augment_image(face_img)
                    for i, augmented_face in enumerate(augmented_faces):
                        cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}_aug{i+1}.jpg", augmented_face, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # Draw rectangle around the face for UI feedback
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Facial Recognition', img)

            # Exit the loop when 'q' is pressed or 60 images (with augmentation) have been saved
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum >= 60:  # Stop after collecting 60 original images (plus augmentations)
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save student details to CSV
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name])

        acknowledgment_label.configure(text="Dataset Created...training in progress")
        threading.Thread(target=play_dataset_creation_sound).start()

        # Train the model with the augmented dataset
        TrainImages()

    else:
        acknowledgment_label.configure(text="Enter valid ID and Name")

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    threading.Thread(target=play_dataset_trained_sound).start()
    acknowledgment_label.configure(text="Dataset Trained Successfully")
    

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


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
                        ts = time.time()
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

# Create buttons
takeImg = tk.Button(window, text="Register", command=TakeImagesWithAugmentation, fg="white", bg="grey", width=12, height=1, activebackground="aqua", font=('times', 15, ' bold '))
takeImg.place(x=50, y=280)

trackImg = tk.Button(window, text="Mark Attendance", command=TrackImages, fg="white", bg="green", width=15, height=1, activebackground="lime", font=('times', 15, ' bold '))
trackImg.place(x=210, y=280)

# Start the Tkinter event loop
window.mainloop()
