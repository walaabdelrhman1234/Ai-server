import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import firebase_admin

from firebase_admin import credentials, firestore

# ============================================
# FIREBASE INIT
# ============================================

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("models/svm_eye_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print(" MODEL LOADED")

# ============================================
# MEDIAPIPE
# ============================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print(" MEDIAPIPE READY")

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" CAMERA ERROR")
    exit()

print(" SYSTEM RUNNING")

# ============================================
# SETTINGS
# ============================================

MODEL_THRESHOLD = 0.85
DEVIATION_THRESHOLD = 0.08
CALIBRATION_FRAMES = 30
SEND_INTERVAL = 5

# ============================================
# MEMORY
# ============================================

baseline_gaze = []
baseline_mean = None
calibrated = False

counter = 0
last_sent_time = 0
warning_time = None
missing_counter = 0

# ============================================
# LANDMARKS
# ============================================

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
LEFT_IRIS = 468
RIGHT_IRIS = 473

# ============================================
# MAIN LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    suspicious = False

    probability = 0.0

    deviation = 0.0

    # ============================================
    # FACE DETECTION
    # ============================================

    if results.multi_face_landmarks:

        missing_counter = 0

        lm = results.multi_face_landmarks[0].landmark

        l_out, l_in, l_top, l_bottom = [lm[i] for i in LEFT_EYE]
        r_out, r_in, r_top, r_bottom = [lm[i] for i in RIGHT_EYE]

        l_pupil = lm[LEFT_IRIS]
        r_pupil = lm[RIGHT_IRIS]

        # ============================================
        # GAZE X
        # ============================================

        left_width = abs(l_in.x - l_out.x)
        right_width = abs(r_in.x - r_out.x)

        gaze_x_left = (l_pupil.x - l_out.x) / left_width if left_width != 0 else 0.5
        gaze_x_right = (r_pupil.x - r_out.x) / right_width if right_width != 0 else 0.5

        gaze_x = (gaze_x_left + gaze_x_right) / 2

        # ============================================
        # GAZE Y
        # ============================================

        left_height = abs(l_top.y - l_bottom.y)
        right_height = abs(r_top.y - r_bottom.y)

        gaze_y_left = (l_pupil.y - l_top.y) / left_height if left_height != 0 else 0.5
        gaze_y_right = (r_pupil.y - r_top.y) / right_height if right_height != 0 else 0.5

        gaze_y = (gaze_y_left + gaze_y_right) / 2

        # ============================================
        # CALIBRATION PHASE
        # ============================================

        if not calibrated:

            baseline_gaze.append([gaze_x, gaze_y])

            cv2.putText(frame,
                        "CALIBRATING...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2)

            cv2.putText(frame,
                        f"Samples: {len(baseline_gaze)}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)

            if len(baseline_gaze) >= CALIBRATION_FRAMES:

                baseline_mean = np.mean(baseline_gaze, axis=0)

                calibrated = True

                print(" CALIBRATION DONE:", baseline_mean)

            cv2.imshow("AI System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        # ============================================
        # FEATURES
        # ============================================

        features = np.array([[
            np.mean([gaze_x, gaze_y]),
            np.max([gaze_x, gaze_y]),
            np.std([gaze_x, gaze_y])
        ]])

        features_scaled = scaler.transform(features)

        probability = model.predict_proba(features_scaled)[0][1]

        # ============================================
        # DEVIATION FROM BASELINE
        # ============================================

        deviation = np.linalg.norm(
            np.array([gaze_x, gaze_y]) - baseline_mean
        )

        print(" FEATURES:", features)
        print(" DEVIATION:", deviation)
        print(" PROB:", probability)

        # ============================================
        # BLINK FILTER
        # ============================================

        blink = False

        if left_height < 0.01 or right_height < 0.01:
            blink = True

        # ============================================
        # HYBRID DECISION
        # ============================================

        if not blink:

            if probability > MODEL_THRESHOLD and deviation > DEVIATION_THRESHOLD:
                suspicious = True
            elif deviation > 0.12:
                suspicious = True
            else:
                suspicious = False

    else:

        missing_counter += 1

        if missing_counter > 2:
            suspicious = True
            probability = 0.95

    # ============================================
    # COUNTER SYSTEM
    # ============================================

    if suspicious:
        counter += 1
    else:
        counter = max(0, counter - 1)

    # ============================================
    # FIREBASE ALERT
    # ============================================

    if counter >= 3:

        counter = 0

        if time.time() - last_sent_time > SEND_INTERVAL:

            db.collection("alerts").add({

                "studentId": "student_test",
                "type": "eye_tracking",
                "confidence": float(probability),
                "deviation": float(deviation),
                "timestamp": time.time()

            })

            last_sent_time = time.time()

    # ============================================
    # UI COLOR
    # ============================================

    color = (0, 255, 0)
    status = "NORMAL"

    if suspicious:
        color = (0, 0, 255)
        status = "CHEATING"

    # ============================================
    # DRAW UI
    # ============================================

    cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)

    cv2.putText(frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    cv2.putText(frame,
                f"AI: {probability:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.putText(frame,
                f"DEV: {deviation:.3f}",
                
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)

    # ============================================
    # SHOW
    # ============================================

    cv2.imshow("Eye Tracking AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()

print(" SYSTEM STOPPED")