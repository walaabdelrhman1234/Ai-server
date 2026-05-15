from flask import Flask, request, jsonify

import cv2
import mediapipe as mp
import numpy as np
import base64
import joblib
import time

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("models/svm_eye_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print(" MODEL LOADED")

# ============================================
# FLASK
# ============================================

app = Flask(__name__)

# ============================================
# MEDIAPIPE
# ============================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print(" MEDIAPIPE READY")

# ============================================
# SETTINGS
# ============================================

MODEL_THRESHOLD = 0.85
DEVIATION_THRESHOLD = 0.08

# ============================================
# MEMORY
# ============================================

baseline = []
baseline_mean = None
calibrated = False

missing_counter = 0

# ============================================
# LANDMARKS
# ============================================

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

LEFT_IRIS = 468
RIGHT_IRIS = 473

# ============================================
# ROUTE
# ============================================

@app.route("/analyze", methods=["POST"])
def analyze():

    global baseline
    global baseline_mean
    global calibrated
    global missing_counter

    try:

        data = request.json

        if "image" not in data:
            return jsonify({
                "status": "error"
            })

        image_data = data["image"]

        image_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "status": "error"
            })

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        suspicious = False
        probability = 0.0
        deviation = 0.0

        # ============================================
        # FACE FOUND
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

            gaze_x_left = (
                (l_pupil.x - l_out.x) / left_width
                if left_width != 0 else 0.5
            )

            gaze_x_right = (
                (r_pupil.x - r_out.x) / right_width
                if right_width != 0 else 0.5
            )

            gaze_x = (gaze_x_left + gaze_x_right) / 2

            # ============================================
            # GAZE Y
            # ============================================

            left_height = abs(l_top.y - l_bottom.y)
            right_height = abs(r_top.y - r_bottom.y)

            gaze_y_left = (
                (l_pupil.y - l_top.y) / left_height
                if left_height != 0 else 0.5
            )

            gaze_y_right = (
                (r_pupil.y - r_top.y) / right_height
                if right_height != 0 else 0.5
            )

            gaze_y = (gaze_y_left + gaze_y_right) / 2

            # ============================================
            # CALIBRATION
            # ============================================

            if not calibrated:

                baseline.append([gaze_x, gaze_y])

                if len(baseline) >= 20:

                    baseline_mean = np.mean(
                        baseline,
                        axis=0
                    )

                    calibrated = True

                    print(" CALIBRATION DONE")

                return jsonify({
                    "status": "calibrating",
                    "counter": len(baseline)
                })

            # ============================================
            # FEATURES
            # ============================================

            features = np.array([[

                np.mean([gaze_x, gaze_y]),
                np.max([gaze_x, gaze_y]),
                np.std([gaze_x, gaze_y])

            ]])

            features_scaled = scaler.transform(features)

            probability = model.predict_proba(
                features_scaled
            )[0][1]

            # ============================================
            # DEVIATION
            # ============================================

            deviation = np.linalg.norm(
                np.array([gaze_x, gaze_y]) - baseline_mean
            )

            print(" FEATURES:", features)
            print(" PROB:", probability)
            print(" DEV:", deviation)

            # ============================================
            # BLINK FILTER
            # ============================================

            blink = False

            if left_height < 0.01 or right_height < 0.01:
                blink = True

            # ============================================
            # HYBRID AI
            # ============================================

            if not blink:

                if (
                    probability > MODEL_THRESHOLD and
                    deviation > DEVIATION_THRESHOLD
                ):
                    suspicious = True

                elif deviation > 0.12:
                    suspicious = True

        else:

            missing_counter += 1

            if missing_counter > 2:
                suspicious = True
                probability = 0.95

        # ============================================
        # RESULT
        # ============================================

        return jsonify({

            "status": "success",
            "cheating": suspicious,
            "probability": float(probability),
            "deviation": float(deviation)

        })

    except Exception as e:

        print(" SERVER ERROR:", e)

        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ============================================
# RUN
# ============================================

if __name__ == "__main__":

    print(" HYBRID SERVER RUNNING")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )