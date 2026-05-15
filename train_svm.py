import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

import joblib

# ============================================
# LOAD DATA
# ============================================

X = np.load("X_flow.npy")

y_original = np.load("y_flow.npy")

print(" X SHAPE:", X.shape)

# ============================================
# CREATE BINARY LABELS
# ============================================

# 0 = NORMAL
# 1 = CHEATING

y = np.where(y_original == 0, 0, 1)

print(" LABELS:", np.unique(y))

# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

print(" TRAIN:", X_train.shape)

print(" TEST:", X_test.shape)

# ============================================
# SCALING
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print(" SCALING COMPLETE")

# ============================================
# TRAIN SVM MODEL
# ============================================

model = SVC(

    kernel='rbf',

    C=1.0,

    probability=True
)

model.fit(X_train, y_train)

print(" AI MODEL TRAINED SUCCESSFULLY")

# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(X_test)

# ============================================
# EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(" MODEL EVALUATION")
print("==============================")

print(f" ACCURACY: {accuracy:.4f}")

print("\n CLASSIFICATION REPORT:\n")

print(classification_report(y_test, y_pred))

print("\n CONFUSION MATRIX:\n")

print(confusion_matrix(y_test, y_pred))

# ============================================
# TEST PROBABILITY
# ============================================

sample = X_test[0].reshape(1, -1)

probability = model.predict_proba(sample)[0][1]

print("\n SAMPLE CHEATING PROBABILITY:")

print(probability)

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(

    model,

    "models/svm_eye_model.pkl"
)

joblib.dump(

    scaler,

    "models/scaler.pkl"
)

print("\n MODEL SAVED SUCCESSFULLY")

print(" READY FOR AI SERVER")