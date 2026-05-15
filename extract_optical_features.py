import os
import cv2
import numpy as np

flow_path = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\optical_flow_results"

X = []
y = []

for person_id in range(15):
    person_folder = f"p{person_id:02d}"
    person_path = os.path.join(flow_path, person_folder)

    if not os.path.exists(person_path):
        continue

    for day in os.listdir(person_path):
        day_path = os.path.join(person_path, day)

        for img_file in os.listdir(day_path):
            img_path = os.path.join(day_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = img / 255.0  # normalization

            mean_val = np.mean(img)
            max_val = np.max(img)
            std_val = np.std(img)

            X.append([mean_val, max_val, std_val])
            y.append(person_id)

X = np.array(X)
y = np.array(y)

np.save("X_flow.npy", X)
np.save("y_flow.npy", y)

print(" Optical Flow features extracted!")