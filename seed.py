# ==============================
# Firebase Seeder (Full Setup)
# ==============================

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta

# ===== Initialize Firebase =====
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ================= USERS =================
def seed_users():
    print(" Seeding Users...")

    db.collection("users").document("admin1").set({
        "name": "Admin",
        "email": "admin@test.com",
        "role": "admin",
        "isActive": True
    })

    db.collection("users").document("student1").set({
        "name": "Shamose",
        "email": "student@test.com",
        "role": "student",
        "isActive": True
    })


# ================= SESSIONS =================
def seed_sessions():
    print(" Seeding Sessions...")

    db.collection("sessions").document("Exam2").set({
        "title": "Computer Exam",
        "createdBy": "admin1",
        "startTime": datetime.now(),
        "endTime": datetime.now() + timedelta(hours=2),
        "status": "active"
    })


# ================= LIVE STUDENTS =================
def seed_live_students():
    print("📡 Seeding Live Students...")

    db.collection("live_students").document("student1").set({
        "name": "Shamose",
        "sessionId": "Exam2",
        "status": "normal",
        "level": "green",
        "cheatingCount": 0,
        "lastUpdate": datetime.now()
    })


# ================= ALERTS =================
def seed_alerts():
    print(" Seeding Alerts...")

    db.collection("alerts").add({
        "studentId": "student1",
        "studentName": "Shamose",
        "sessionId": "Exam2",
        "type": "eye_tracking",
        "confidence": 0.92,
        "level": "yellow",
        "message": "Looking away detected",
        "timestamp": datetime.now()
    })


# ================= CHEATING LOGS =================
def seed_logs():
    print(" Seeding Logs...")

    db.collection("cheating_logs").add({
        "studentId": "student1",
        "sessionId": "Exam2",
        "event": "looking_left",
        "confidence": 0.92,
        "timestamp": datetime.now()
    })


# ================= RUN ALL =================
def run_all():
    seed_users()
    seed_sessions()
    seed_live_students()
    seed_alerts()
    seed_logs()

    print(" Firebase Setup Completed Successfully!")


# ===== RUN =====
if __name__ == "__main__":
    run_all()