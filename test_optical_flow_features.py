import os
import cv2
import numpy as np
import pandas as pd
import joblib

# ===== مسارات الملفات =====
images_folder = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\images"
optical_flow_folder = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\optical_flow_results"  # لو عندك صور Optical Flow لكل صورة
svm_model_path = r"C:\Users\/.Lenovo\Desktop\eye_tracking_ai\svm_eye_model.pkl"
scaler_path = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\scaler.pkl"
results_excel = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\test_results_optflow.xlsx"

# ===== تحميل الموديل والـ scaler =====
svm_model = joblib.load(svm_model_path)
scaler = joblib.load(scaler_path)

# ===== لو عندك التسميات الصحيحة لكل صورة =====
# correct_labels = {"image_1.jpg": 3, "image_2.jpg": 0, ...}
correct_labels = None  # خليها None لو ما عندك

# ===== القوائم =====
images_list = []
predicted_list = []

# ===== توقع كل صورة باستخدام خصائص Optical Flow =====
for img_file in os.listdir(images_folder):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # مسار الصورة العادية
    img_path = os.path.join(images_folder, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    # تصغير الصورة لو كبيرة
    img = cv2.resize(img, (320, 240))

    # ===== لو عندك Optical Flow للصور، ممكن تحمليه =====
    # افترضنا إن كل صورة لها Flow في نفس الترتيب داخل optical_flow_folder
    flow_path = os.path.join(optical_flow_folder, img_file)
    if os.path.exists(flow_path):
        flow_img = cv2.imread(flow_path, cv2.IMREAD_GRAYSCALE)
        flow_img = flow_img / 255.0
    else:
        # لو ما عندك Flow، ممكن نستخدم الصورة الرمادية العادية
        flow_img = img / 255.0

    # ===== استخراج الخصائص زي التدريب =====
    mean_val = np.mean(flow_img)
    max_val = np.max(flow_img)
    std_val = np.std(flow_img)

    X = np.array([[mean_val, max_val, std_val]])
    X_scaled = scaler.transform(X)

    pred = svm_model.predict(X_scaled)[0]

    images_list.append(img_file)
    predicted_list.append(pred)

# ===== إنشاء DataFrame =====
df = pd.DataFrame({
    "Image": images_list,
    "Predicted_person": predicted_list
})

# ===== حساب الدقة لو عندك التسميات الصحيحة =====
if correct_labels:
    df['Correct'] = df.apply(lambda row: correct_labels.get(row['Image'], -1) == row['Predicted_person'], axis=1)
    accuracy = df['Correct'].mean() * 100
    print(f" الدقة الإجمالية: {accuracy:.2f}%")

# ===== حفظ النتائج =====
df.to_excel(results_excel, index=False)
print(f" تم حفظ النتائج في: {results_excel}")

# ===== تحليل سريع =====
counts = df['Predicted_person'].value_counts().sort_index()
print("\n عدد الصور المتوقع لكل شخص:")
print(counts)