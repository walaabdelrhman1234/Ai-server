import pandas as pd
import os

# ===== مسار ملف النتائج من التست =====
results_excel = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\test_results_optflow.xlsx"

# ===== تحميل النتائج =====
df = pd.read_excel(results_excel)

correct_labels = {}

for i in range(1, 801):  # 0001.jpg → 0800.jpg
    img_name = f"{i:04d}.jpg"
   
    correct_labels[img_name] = 0  

# ===== إضافة عمود 'Correct' =====
df['Correct'] = df['Image'].apply(lambda x: int(df['Predicted_person'][df['Image'] == x].values[0] == correct_labels.get(x, -1)))

# ===== حساب الدقة لكل شخص =====
accuracy_per_person = {}

for person_id in set(correct_labels.values()):
    person_images = [img for img, pid in correct_labels.items() if pid == person_id]
    correct_count = df[df['Image'].isin(person_images) & (df['Predicted_person'] == person_id)].shape[0]
    total_count = len(person_images)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    accuracy_per_person[person_id] = accuracy

# ===== طباعة النتائج =====
print("  دقة كل شخص (%):")
for pid, acc in accuracy_per_person.items():
    print(f"Person {pid}: {acc:.2f}%")

# ===== حفظ نسخة تحليلية جديدة =====
output_file = r"C:\Users\Lenovo\Desktop\eye_tracking_ai\test_results_accuracy.xlsx"
df.to_excel(output_file, index=False)
print(f"\n تم حفظ نسخة تحليلية مع الدقة في: {output_file}")