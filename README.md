# Student Performance Predictor (FNN)

Deep learning project for the module **Python and Fundamentals of AI (ITLPA701)**, Rwanda Polytechnic, Ngoma College.

## Problem
Predict whether a student will **pass or fail** the final exam (final grade G3 of 10 or more out of 20), so teachers can give extra support to students at risk.

## Dataset
UCI Student Performance dataset, math course file (`student-mat.csv`): 395 students, 33 columns, no missing values, no duplicates.
Source: https://archive.ics.uci.edu/dataset/320/student+performance

## Approach
A **Feed-forward Neural Network (FNN)** built with TensorFlow/Keras, chosen because the data is a table of numbers with two classes.

- Preprocessing: pass/fail target, encoding of text columns, stratified split (252 train / 64 validation / 79 test), StandardScaler fitted on training data only
- Engineered features: ParentsEdu, AlcoholIndex, LogAbsences, GradeAvg, GradeTrend (19 inputs in total)
- Model: Dense layers with ReLU, dropout, sigmoid output, Adam optimiser, early stopping, class weights
- Tuning: 4 configurations compared on the validation set; best was 16-8 units, dropout 0.2, learning rate 0.001

## Results (test set, 79 students)
| Metric | Value |
|---|---|
| Accuracy | 84.8% |
| Precision (Pass) | 91.8% |
| Recall (Pass) | 84.9% |
| F1-score (Pass) | 88.2% |
| Recall (Fail) | 84.6% (22 of 26 failing students found) |

Without the first and second term grades (G1, G2), accuracy dropped to about 65.8%, so earlier grades are the strongest predictors.

## Project files
- `student_performance_fnn.py`: the full program (run cell by cell or all at once)
- `student-mat.csv`: the dataset
- `saved_model/`: trained model (`student_fnn.keras`), scaler, `config.json`, `report.txt` and charts
- `Student_Performance_CAT.pptx`: presentation

## How to run
1. Install Python 3.12 (TensorFlow does not support Python 3.14 yet)
2. Create an environment and install the libraries:
```
   py -3.12 -m venv venv
   venv\Scripts\python -m pip install tensorflow pandas numpy scikit-learn matplotlib seaborn joblib gradio
```
3. Run the program:
```
   venv\Scripts\python student_performance_fnn.py
```
4. Close each chart window as it appears. The Gradio demo then opens at http://127.0.0.1:7860

## Responsible use
- The prediction is decision support only; a teacher makes the final decision.
- Sex, home area and family support could cause unfair predictions, so results should be audited between groups.
- The data covers 395 Portuguese students, so the model should be retrained on local data before real use.
- Student data is personal and must be collected with consent and stored securely.