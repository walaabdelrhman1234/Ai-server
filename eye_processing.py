import os
import cv2


base_path = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\dataset\MPIIGaze\Data\Original"

save_base_path = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\processed_images"


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# لوب على كل الأشخاص p00 إلى p14
for person_id in range(15):
    person_folder = f"p{person_id:02d}"
    person_path = os.path.join(base_path, person_folder)
    save_person_path = os.path.join(save_base_path, person_folder)
    os.makedirs(save_person_path, exist_ok=True)
    
    if not os.path.exists(person_path):
        print(f"{person_folder} غير موجود، بيتجاهل")
        continue
    
    # لوب على كل الأيام داخل فولدر الشخص
    day_folders = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
    if not day_folders:
        print(f"{person_folder} ما فيه أي يوم، بيتجاهل")
        continue
    
    for day_folder in day_folders:
        day_path = os.path.join(person_path, day_folder)
        save_day_path = os.path.join(save_person_path, day_folder)
        os.makedirs(save_day_path, exist_ok=True)
        
        # كل الصور داخل فولدر اليوم
        image_files = [f for f in os.listdir(day_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            print(f"مافي صور في {day_path}, بيتجاهل")
            continue
        
        for img_file in image_files:
            img_path = os.path.join(day_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Preprocessing: رمادي + إزالة ضوضاء
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detection: تحديد العينين
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # حفظ الصورة المعالجة
            save_path = os.path.join(save_day_path, img_file)
            cv2.imwrite(save_path, img)

print(" تمت معالجة كل الصور الموجودة وحفظها في processed_images!")