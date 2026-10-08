# =============================================================================
# STUDENT PERFORMANCE PREDICTION - Feed-forward Neural Network (FNN)
# Module: Python and Fundamentals of AI (ITLPA701)
#
# Problem : Predict if a student will PASS or FAIL the final exam (G3 >= 10 of 20)
#           so teachers can give early support to students at risk.
# Dataset : UCI "Student Performance" (student-mat.csv, 395 students)
#           https://archive.ics.uci.edu/dataset/320/student+performance
# =============================================================================

# %% 1. ENVIRONMENT SETUP  (environment configured, functionalities specified)
# FUNCTIONALITIES OF THE APPLICATION
#   1. Load the student dataset
#   2. Clean and preprocess the data (missing values, duplicates, encoding, scaling)
#   3. Engineer new features (parents' education, alcohol index, grade trend)
#   4. Build and train an FNN
#   5. Evaluate (accuracy, precision, recall, F1, confusion matrix)
#   6. Tune parameters using validation results
#   7. Save model, scaler and configuration
#   8. Predict through a Gradio web interface

import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Same seed = same results every run (reproducibility)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "student-mat.csv"
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# True  = also use first and second term grades (G1, G2) -> higher accuracy
# False = use only background information (earlier warning, but harder)
USE_PREVIOUS_GRADES = True

print("TensorFlow version:", tf.__version__)

# %% 2. DATA ACQUISITION  (data acquired from the given source)
# sep=None lets pandas detect ";" (UCI file) or "," (Kaggle copy) automatically
df = pd.read_csv(DATA_PATH, sep=None, engine="python")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nFinal grade G3 summary:\n", df["G3"].describe())

# %% 3. DATA PREPROCESSING  (data pre-processed)
data = df.copy()

# 3.1 Make sure grade columns are numbers, remove duplicates, fill missing values
for col in ["G1", "G2", "G3"]:
    data[col] = pd.to_numeric(data[col])

data = data.drop_duplicates()
for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

# 3.2 Target: PASS (1) if final grade G3 >= 10, otherwise FAIL (0)
data["Pass"] = (data["G3"] >= 10).astype(int)
print("\nPass/Fail counts:\n", data["Pass"].value_counts())

# 3.3 Feature selection: keep columns that are meaningful and easy to explain.
#     G3 is removed because it IS the answer (it would be cheating).
SELECTED = ["sex", "age", "address", "Medu", "Fedu", "studytime", "failures",
            "schoolsup", "famsup", "higher", "internet", "romantic",
            "goout", "Dalc", "Walc", "health", "absences"]
if USE_PREVIOUS_GRADES:
    SELECTED += ["G1", "G2"]
data = data[SELECTED + ["Pass"]]

# 3.4 Encode text columns as numbers (the network only accepts numbers)
ENCODING = {
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},          # Rural / Urban
    "schoolsup": {"no": 0, "yes": 1},     # extra school support
    "famsup": {"no": 0, "yes": 1},        # family support
    "higher": {"no": 0, "yes": 1},        # wants higher education
    "internet": {"no": 0, "yes": 1},      # internet at home
    "romantic": {"no": 0, "yes": 1},      # in a relationship
}
for col, mapping in ENCODING.items():
    data[col] = data[col].map(mapping)

# %% 4. FEATURE ENGINEERING  (data features + model features engineered)
# One function used for BOTH training and prediction, so the web app builds
# features in exactly the same way as the training code.

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    # Combined education level of both parents (0-8)
    f["ParentsEdu"] = f["Medu"] + f["Fedu"]
    # Average alcohol consumption (weekday and weekend)
    f["AlcoholIndex"] = (f["Dalc"] + f["Walc"]) / 2
    # Absences are very skewed (a few students miss many days) -> log reduces it
    f["LogAbsences"] = np.log1p(f["absences"])
    if "G1" in f.columns and "G2" in f.columns:
        f["GradeAvg"] = (f["G1"] + f["G2"]) / 2     # overall level
        f["GradeTrend"] = f["G2"] - f["G1"]         # improving or declining?
    return f.drop(columns=["Medu", "Fedu", "Dalc", "Walc", "absences"])

X_df = engineer_features(data.drop(columns=["Pass"]))
y = data["Pass"].values
FEATURE_COLUMNS = list(X_df.columns)
print("Input features used by the model (%d):" % len(FEATURE_COLUMNS), FEATURE_COLUMNS)

# Correlation with passing (supports and justifies the chosen features)
corr = X_df.assign(Pass=y).corr()["Pass"].drop("Pass")
print("\nCorrelation with passing:\n", corr.sort_values(key=abs, ascending=False).round(3))

plt.figure(figsize=(6, 5))
corr.sort_values().plot(kind="barh", color="steelblue")
plt.title("Correlation of each feature with passing")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_correlation.png", dpi=120)
plt.show()

# %% 5. SPLIT AND SCALE
# Stratify keeps the same pass/fail ratio in every subset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_df.values, y, test_size=0.2, stratify=y, random_state=SEED)

# Validation set (taken from training data) is used for tuning; test set is used once
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED)

# Fit the scaler on TRAINING data only (prevents data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# More students pass than fail, so failing students get a higher weight
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
print("\nClass weights:", class_weight)
print("Train / validation / test sizes:", len(X_train), len(X_val), len(X_test))

# %% 6. MODEL DESIGN  (approach justified, parameters selected and explained)
# WHY AN FNN?  The data is structured (a table of numbers) and the answer is
#   two classes (pass / fail), so a feed-forward network with a sigmoid output
#   is a suitable deep learning model.
# PARAMETERS
#   units 32 -> 16 : enough capacity for ~19 features; small because the dataset is small
#   ReLU           : fast and avoids vanishing gradients
#   Dropout        : randomly switches off neurons so the model does not memorise
#   Sigmoid output : gives probability of passing between 0 and 1
#   Adam, lr 0.001 : adaptive optimiser with a stable default learning rate
#   binary cross-entropy : standard loss for two-class problems
#   EarlyStopping  : stops training when validation loss stops improving

def build_model(input_dim, units1=32, units2=16, dropout=0.3, lr=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units1, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(units2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(params, epochs=200, batch_size=16):
    model = build_model(X_train.shape[1], **params)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        class_weight=class_weight, callbacks=[stop], verbose=0)
    return model, history

# Show the structure of the network
build_model(X_train.shape[1]).summary()


# %% 7. TUNING  (parameters adjusted based on evaluation results)
# Train several configurations and choose the best using the VALIDATION set
configs = {
    "Baseline (32-16, drop 0.3, lr 0.001)": dict(units1=32, units2=16, dropout=0.3, lr=0.001),
    "Smaller (16-8, drop 0.2, lr 0.001)":   dict(units1=16, units2=8, dropout=0.2, lr=0.001),
    "Larger (64-32, drop 0.5, lr 0.001)":   dict(units1=64, units2=32, dropout=0.5, lr=0.001),
    "Slow LR (32-16, drop 0.3, lr 0.0005)": dict(units1=32, units2=16, dropout=0.3, lr=0.0005),
}

rows, trained = [], {}
for name, params in configs.items():
    model, history = train_model(params)
    val_pred = (model.predict(X_val, verbose=0).ravel() > 0.5).astype(int)
    rows.append({"Configuration": name,
                 "Val accuracy": round(accuracy_score(y_val, val_pred), 3),
                 "Val F1 (macro)": round(f1_score(y_val, val_pred, average="macro"), 3),
                 "Epochs run": len(history.history["loss"])})
    trained[name] = (model, history, params)

tuning_table = pd.DataFrame(rows)
print("\nTUNING RESULTS")
print(tuning_table.to_string(index=False))

best_name = tuning_table.sort_values("Val F1 (macro)", ascending=False).iloc[0]["Configuration"]
model, history, best_params = trained[best_name]
print("\nBest configuration:", best_name)

# %% 8. TRAINING CURVES
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/training_curves.png", dpi=120)
plt.show()

# %% 9. FINAL EVALUATION ON THE TEST SET  (evaluation implemented and interpreted)
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fail_recall = recall_score(y_test, y_pred, pos_label=0)   # at-risk students found

print("\nFINAL TEST RESULTS")
print(f"Accuracy           : {acc:.3f}")
print(f"Precision (Pass)   : {prec:.3f}")
print(f"Recall (Pass)      : {rec:.3f}")
print(f"F1-score (Pass)    : {f1:.3f}")
print(f"Recall (Fail)      : {fail_recall:.3f}   <- share of at-risk students we caught")
print("\n", classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/confusion_matrix.png", dpi=120)
plt.show()

# HOW TO INTERPRET (use your own numbers when you explain):
#  - Accuracy : share of all students predicted correctly.
#  - Precision (Pass): of the students predicted to pass, how many really passed.
#  - Recall (Pass)   : of the students who really passed, how many we identified.
#  - Recall (Fail)   : of the students who really failed, how many we warned.
#                      This is the MOST important number here, because a missed
#                      failing student gets no extra help.
#  - Confusion matrix: bottom-left = failing students wrongly predicted to pass (dangerous);
#                      top-right   = passing students wrongly flagged (only extra attention).
# %% 9b. APPLICATION DESIGN / ARCHITECTURE  (Task 4)
fig, ax = plt.subplots(figsize=(14, 3.8))
ax.axis("off")
boxes = [
    ("INPUT\nStudent form (Gradio):\nage, failures,\nstudy time, G1, G2", "#cfe8ff"),
    ("PREPROCESSING\nEncode yes/no, M/F,\nU/R as numbers", "#fff2cc"),
    ("FEATURE ENGINEERING\nParentsEdu,\nAlcoholIndex,\nLogAbsences,\nGradeAvg, GradeTrend", "#fff2cc"),
    ("SCALING\nStandardScaler\n(fitted on\ntraining data)", "#fff2cc"),
    (f"AI MODEL (FNN)\nDense {best_params['units1']} -> Dense {best_params['units2']}\nDropout, sigmoid\nbest of 4 configs", "#d9ead3"),
    ("OUTPUT\nLIKELY TO PASS\nor AT RISK\n+ probability", "#f4cccc"),
]
n_boxes = len(boxes)
w, gap = 2.0, 0.4
for i, (text, color) in enumerate(boxes):
    x = i * (w + gap)
    ax.add_patch(plt.Rectangle((x, 0.2), w, 2.4, facecolor=color, edgecolor="black"))
    ax.text(x + w / 2, 1.4, text, ha="center", va="center", fontsize=8)
    if i < n_boxes - 1:
        ax.annotate("", xy=(x + w + gap, 1.4), xytext=(x + w, 1.4),
                    arrowprops=dict(arrowstyle="->", lw=1.8))
ax.set_xlim(-0.1, n_boxes * (w + gap))
ax.set_ylim(0, 2.9)
ax.set_title("Application architecture: Student Performance Predictor", fontsize=11)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/architecture.png", dpi=130)
plt.show()

# %% 9c. INTERPRETATION OF RESULTS  (written automatically from your numbers)
n_fail = int(cm[0].sum())     # students who really failed in the test set
caught = int(cm[0, 0])        # failing students the model found
interpretation = f"""
INTERPRETATION OF RESULTS
- Accuracy {acc:.1%}: out of 100 students the model has never seen, about {acc*100:.0f} are predicted correctly.
- Recall (Fail) {fail_recall:.1%}: the model found {caught} of the {n_fail} students who really failed.
  This is the most important number, because a missed failing student gets no extra help.
- Precision (Pass) {prec:.1%}: when the model says "will pass", it is right about {prec*100:.0f} times out of 100.
- Recall (Pass) {rec:.1%} and F1-score {f1:.1%}: passing students are also identified well.
- Confusion matrix: a failing student predicted to pass is the dangerous error.
  A passing student flagged as at risk only receives extra attention.
- Tuning: four configurations were compared on the validation set. The best was "{best_name}".
  Smaller networks suit this small dataset ({len(X_train)} training students) because they overfit less.
- Experiment: without the earlier grades G1 and G2 the accuracy dropped (see section 13), so earlier grades are the strongest predictors.
- Conclusion: the model is a useful early-warning aid, but about {(1-acc)*100:.0f}% of predictions are still wrong,
  so a teacher must make the final decision.
"""
print(interpretation)

# %% 9d. JUSTIFICATION OF APPROACH AND PARAMETERS  (written automatically)
justification = f"""
JUSTIFICATION OF THE APPROACH AND PARAMETERS
- Problem: identify students at risk of failing early, so teachers can give support.
- Approach: Deep Learning with an FNN. The data is a table of numbers with a pass/fail answer.
  A CNN is made for images and an RNN for sequences, so they do not fit this data.
- Units {best_params['units1']} and {best_params['units2']}: a small network, because the dataset is small.
- Dropout {best_params['dropout']}: switches off random neurons so the model does not memorise.
- Learning rate {best_params['lr']}: the step size of the Adam optimiser; small steps give stable learning.
- Batch size 16: small batches suit a small dataset and give more updates.
- Early stopping (patience 15): training stops when validation loss stops improving.
- Class weights: failing students count more, because fewer students fail.
- Decision threshold 0.5: a student with probability above 0.5 is predicted to pass.
"""
print(justification)

with open(f"{SAVE_DIR}/report.txt", "w", encoding="utf-8") as fh:
    fh.write(interpretation + "\n" + justification)
print("Report saved to:", os.path.abspath(f"{SAVE_DIR}/report.txt"))



# %% 10. SAVE EVERYTHING  (model/configuration saved so it can be reproduced)
model.save(f"{SAVE_DIR}/student_fnn.keras")
joblib.dump(scaler, f"{SAVE_DIR}/scaler.pkl")

config = {
    "seed": SEED,
    "use_previous_grades": USE_PREVIOUS_GRADES,
    "feature_columns": FEATURE_COLUMNS,
    "best_configuration": best_name,
    "hyperparameters": best_params,
    "optimizer": "Adam",
    "loss": "binary_crossentropy",
    "batch_size": 16,
    "decision_threshold": 0.5,
    "test_metrics": {"accuracy": round(acc, 4), "precision_pass": round(prec, 4),
                     "recall_pass": round(rec, 4), "f1_pass": round(f1, 4),
                     "recall_fail": round(fail_recall, 4)},
}
with open(f"{SAVE_DIR}/config.json", "w") as fh:
    json.dump(config, fh, indent=2)
print("\nSaved to:", os.path.abspath(SAVE_DIR))

# %% 11. DEMO INTERFACE WITH GRADIO  (application demonstrated)
import gradio as gr

# Load from disk, exactly as a user of the saved application would
app_model = keras.models.load_model(f"{SAVE_DIR}/student_fnn.keras")
app_scaler = joblib.load(f"{SAVE_DIR}/scaler.pkl")
with open(f"{SAVE_DIR}/config.json") as fh:
    saved = json.load(fh)
app_cols = saved["feature_columns"]
app_uses_grades = saved["use_previous_grades"]

def predict_student(sex, age, address, medu, fedu, studytime, failures,
                    schoolsup, famsup, higher, internet, romantic,
                    goout, dalc, walc, health, absences, g1, g2):
    row = {"sex": sex, "age": age, "address": address, "Medu": medu, "Fedu": fedu,
           "studytime": studytime, "failures": failures, "schoolsup": schoolsup,
           "famsup": famsup, "higher": higher, "internet": internet,
           "romantic": romantic, "goout": goout, "Dalc": dalc, "Walc": walc,
           "health": health, "absences": absences}
    if app_uses_grades:
        row["G1"], row["G2"] = g1, g2
    row = pd.DataFrame([row])
    # Same encoding and feature engineering as in training
    for col, mapping in ENCODING.items():
        row[col] = row[col].map(mapping)
    row = engineer_features(row)[app_cols]
    prob = float(app_model.predict(app_scaler.transform(row.values), verbose=0)[0][0])
    if prob > 0.5:
        result = "LIKELY TO PASS"
    else:
        result = "AT RISK OF FAILING - recommend extra support"
    return (f"{result}\nProbability of passing: {prob:.1%}\n\n"
            "Decision-support only: a teacher must make the final judgement.")

demo = gr.Interface(
    fn=predict_student,
    inputs=[
        gr.Dropdown(["F", "M"], value="F", label="Sex"),
        gr.Slider(15, 22, value=17, step=1, label="Age"),
        gr.Dropdown(["U", "R"], value="U", label="Home area (U = urban, R = rural)"),
        gr.Slider(0, 4, value=2, step=1, label="Mother's education (0-4)"),
        gr.Slider(0, 4, value=2, step=1, label="Father's education (0-4)"),
        gr.Slider(1, 4, value=2, step=1, label="Weekly study time (1 = <2h ... 4 = >10h)"),
        gr.Slider(0, 3, value=0, step=1, label="Past class failures"),
        gr.Dropdown(["no", "yes"], value="no", label="Extra school support"),
        gr.Dropdown(["no", "yes"], value="yes", label="Family support"),
        gr.Dropdown(["yes", "no"], value="yes", label="Wants higher education"),
        gr.Dropdown(["yes", "no"], value="yes", label="Internet at home"),
        gr.Dropdown(["no", "yes"], value="no", label="In a relationship"),
        gr.Slider(1, 5, value=3, step=1, label="Going out with friends (1-5)"),
        gr.Slider(1, 5, value=1, step=1, label="Weekday alcohol (1-5)"),
        gr.Slider(1, 5, value=1, step=1, label="Weekend alcohol (1-5)"),
        gr.Slider(1, 5, value=4, step=1, label="Health (1 = bad, 5 = very good)"),
        gr.Slider(0, 75, value=4, step=1, label="Number of absences"),
        gr.Slider(0, 20, value=11, step=1, label="First term grade G1 (0-20)"),
        gr.Slider(0, 20, value=11, step=1, label="Second term grade G2 (0-20)"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Student Performance Predictor (FNN)",
    description="Enter student information to estimate whether the student will pass the final exam.",
)
demo.launch()


# %% 12. RESPONSIBLE USE
# RISK 1 - Labelling students: a prediction can become a self-fulfilling label
#   or lower a teacher's expectations of a student.
#   Reduction: use it only to offer extra support, never to punish or exclude;
#   a teacher always makes the final decision.
# RISK 2 - Bias and fairness: features such as sex, home area or family support
#   could lead to unfair predictions for some groups.
#   Reduction: remove or audit sensitive features and compare results between groups.
# RISK 3 - Small, foreign data: only 395 students from two Portuguese schools,
#   not representative of Rwandan students.
#   Reduction: retrain on local school data and keep checking performance.
# RISK 4 - Privacy: student data is personal.
#   Reduction: collect with consent, anonymise records, store securely.


# %% 13. EXPERIMENT RESULTS (comparison for tuning)
# Experiment 1: USE_PREVIOUS_GRADES = True  (G1 and G2 included)
#   Accuracy = 0.848   Recall (Fail) = 0.846
# Experiment 2: USE_PREVIOUS_GRADES = False (G1 and G2 removed)
#   Accuracy = 0.658  Recall (Fail) = 0.846
# Conclusion: earlier grades are the strongest predictors (see correlation chart).