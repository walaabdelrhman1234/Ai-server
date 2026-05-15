import os
import cv2
import numpy as np


base_path = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\processed_images"
save_base_path = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\optical_flow_results"

os.makedirs(save_base_path, exist_ok=True)

# عدد الأشخاص
NUM_PERSONS = 15

for person_id in range(NUM_PERSONS):
    person_folder = f"p{person_id:02d}"
    person_path = os.path.join(base_path, person_folder)
    save_person_path = os.path.join(save_base_path, person_folder)
    os.makedirs(save_person_path, exist_ok=True)

    if not os.path.exists(person_path):
        print(f" {person_folder} غير موجود")
        continue

    print(f"\n معالجة الشخص: {person_folder}")

    day_folders = [
        d for d in os.listdir(person_path)
        if os.path.isdir(os.path.join(person_path, d))
    ]

    if not day_folders:
        print(f" {person_folder} ما عندو أيام")
        continue

    for day_folder in day_folders:
        day_path = os.path.join(person_path, day_folder)
        save_day_path = os.path.join(save_person_path, day_folder)
        os.makedirs(save_day_path, exist_ok=True)

        image_files = sorted([
            f for f in os.listdir(day_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(image_files) < 2:
            print(f" {person_folder}/{day_folder}: أقل من صورتين")
            continue

        prev_gray = None
        flow_count = 0

        for img_file in image_files:
            img_path = os.path.join(day_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f" لم تقرأ الصورة: {img_file}")
                continue

            img = cv2.resize(img, (320, 240))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                save_name = f"flow_{flow_count:04d}.jpg"
                cv2.imwrite(os.path.join(save_day_path, save_name), flow_img)
                flow_count += 1

            prev_gray = gray

        if flow_count == 0:
            print(f" {person_folder}/{day_folder}: ما اتولد Optical Flow")
        else:
            print(f"  {person_folder}/{day_folder}: {flow_count} صورة Flow")

print("\n انتهى حساب Optical Flow بنجاح")