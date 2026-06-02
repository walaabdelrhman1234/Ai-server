import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ================= INIT FIREBASE =================
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# ================= DELETE COLLECTION =================
def delete_collection(collection_name, batch_size=50):
    coll_ref = db.collection(collection_name)
    docs = coll_ref.limit(batch_size).stream()

    deleted = 0

    for doc in docs:
        print(f"Deleting {collection_name}/{doc.id}")
        doc.reference.delete()
        deleted += 1

    if deleted >= batch_size:
        return delete_collection(collection_name, batch_size)

# ================= RESET DATABASE =================
def reset_database():
    collections = [
        "users",
        "sessions",
        "alerts",
        "exams",
        "cheating_logs"
    ]

    for c in collections:
        delete_collection(c)
        print(f"{c} CLEARED")

    print(" ALL COLLECTIONS DELETED")


# ================= REBUILD DATABASE =================
def rebuild_database():

    # USERS
    db.collection("users").document("student1").set({
        "name": "Walaa",
        "email": "walaa@gmail.com",
        "role": "student",
        "isActive": True,
        "faceImageUrl": ""
    })

    db.collection("users").document("admin1").set({
        "name": "Admin",
        "email": "admin@gmail.com",
        "role": "admin",
        "isActive": True,
        "faceImageUrl": ""
    })

    # SESSION
    db.collection("sessions").document("Exam2").set({
        "studentId": "student1",
        "studentName": "Walaa",
        "startTime": datetime.now(),
        "endTime": None,
        "status": "running",
        "cheatingCount": 0,
        "score": 0,
        "duration": 0,
        "faceVerified": True
    })

    # EXAMS
    db.collection("exams").document("exam1").set({
        "title": "AI Test",
        "duration": 30,
        "questions": [
            {
                "question": "What is AI?",
                "options": ["Animal", "AI", "Input", "None"],
                "answer": 1
            },
            {
                "question": "What is Flutter?",
                "options": ["Framework", "DB", "OS", "Game"],
                "answer": 0
            }
        ]
    })

    print(" DATABASE REBUILT SUCCESSFULLY")


# ================= RUN =================
if __name__ == "__main__":
    print(" STARTING SEED SCRIPT...")

    reset_database()
    rebuild_database()

    print(" DONE  FIREBASE IS READY")