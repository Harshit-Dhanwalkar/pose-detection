import mediapipe as mp
import cv2 as cv
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Pose and Drawing Utils
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

# Directory containing images
directory = "images/"
data = []

# Define the points to extract from the pose landmarks
points = [
    mpPose.PoseLandmark.NOSE,
    mpPose.PoseLandmark.LEFT_SHOULDER,
    mpPose.PoseLandmark.RIGHT_SHOULDER,
    # Add more landmarks as needed
]

# Extract the column names for the dataset
column_names = []
for landmark in points:
    x = str(landmark).split('.')[-1]
    column_names.append(x + "_x")
    column_names.append(x + "_y")
    column_names.append(x + "_z")
    column_names.append(x + "_visibility")

df = pd.DataFrame(columns=column_names)

# Process each image in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(directory, filename)
        img = cv.imread(file_path)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            img_height, img_width, _ = img.shape
            annotated_image = img.copy()
            mpDraw.draw_landmarks(annotated_image, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                  mpDraw.DrawingSpec(
                                      color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2))

            landmarks = results.pose_landmarks.landmark
            temp = []
            for landmark in points:
                temp.extend([landmarks[landmark].x, landmarks[landmark].y,
                            landmarks[landmark].z, landmarks[landmark].visibility])

            df.loc[len(df)] = temp

        cv.imshow('Annotated Image', annotated_image)
        cv.waitKey(0)

cv.destroyAllWindows()

# Save the dataset to a CSV file
df.to_csv("dataset.csv", index=False)
