"""
Simulate the exact Streamlit form input and prediction flow from app.py
This reproduces what happens when user clicks "🚀 Predict Diagnosis"
"""
import pandas as pd
import numpy as np
import joblib
import os

assets_dir = 'streamlit_assets'

# Load everything
rf_model = joblib.load(os.path.join(assets_dir, 'rf_model.joblib'))
xgb_model = joblib.load(os.path.join(assets_dir, 'xgb_model.joblib'))
le = joblib.load(os.path.join(assets_dir, 'label_encoder.joblib'))
lb = joblib.load(os.path.join(assets_dir, 'label_binarizer.joblib'))
scaler = joblib.load(os.path.join(assets_dir, 'scaler.joblib'))

feature_columns_for_app = [
    'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels',
    'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy',
    'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments',
    'Digestive Enzyme Levels', 'Birth Weight',
    'Genetic Markers_Positive',
    'Autoantibodies_Positive', 
    'Family History_Yes', 
    'Environmental Factors_Present',
    'Physical Activity_Moderate',
    'Dietary Habits_Unhealthy',
    'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Hispanic', 'Ethnicity_Other',
    'Socioeconomic Factors_High Income', 'Socioeconomic Factors_Upper Class',
    'Smoking Status_Yes',
    'Alcohol Consumption_Moderate',
    'Glucose Tolerance Test_Impaired',
    'History of PCOS_Yes', 
    'Previous Gestational Diabetes_Yes',
    'Pregnancy History_Complications',
    'Cystic Fibrosis Diagnosis_Yes',
    'Steroid Use History_Yes', 
    'Genetic Testing_Positive',
    'Liver Function Tests_Abnormal', 
    'Urine Test_Ketones Present', 'Urine Test_Protein Present',
    'Early Onset Symptoms_Yes'
]

numerical_features_names = [
    'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels',
    'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy',
    'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments',
    'Digestive Enzyme Levels', 'Birth Weight'
]

numerical_features_to_scale = ['Birth Weight']

original_categorical_features_map = {
    'Genetic Markers': ['Positive', 'Negative'],
    'Autoantibodies': ['Positive', 'Negative'],
    'Family History': ['Yes', 'No'],
    'Environmental Factors': ['Present', 'Absent'],
    'Physical Activity': ['High', 'Low', 'Moderate', 'Sedentary'],
    'Dietary Habits': ['Unhealthy', 'Healthy'],
    'Ethnicity': ['Asian', 'Black', 'Hispanic', 'Other', 'White'],
    'Socioeconomic Factors': ['High Income', 'Low Income', 'Middle Class', 'Upper Class'],
    'Smoking Status': ['Yes', 'No'],
    'Alcohol Consumption': ['Heavy', 'Moderate', 'No'],
    'Glucose Tolerance Test': ['Impaired', 'Normal'],
    'History of PCOS': ['Yes', 'No'],
    'Previous Gestational Diabetes': ['Yes', 'No'],
    'Pregnancy History': ['Complications', 'Normal'],
    'Cystic Fibrosis Diagnosis': ['Yes', 'No'],
    'Steroid Use History': ['Yes', 'No'],
    'Genetic Testing': ['Positive', 'Negative'],
    'Liver Function Tests': ['Abnormal', 'Normal'],
    'Urine Test': ['Ketones Present', 'Glucose Present', 'Normal', 'Protein Present'],
    'Early Onset Symptoms': ['Yes', 'No']
}

def preprocess_input(input_df):
    processed_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns_for_app)
    
    for col in numerical_features_names:
        if col in input_df.columns:
            processed_df[col] = input_df[col].values

    for original_col, possible_values in original_categorical_features_map.items():
        if original_col in input_df.columns:
            value = input_df[original_col].iloc[0]
            ohe_col_name = f"{original_col}_{value}"
            if ohe_col_name in processed_df.columns:
                processed_df[ohe_col_name] = 1

    for col_to_scale in numerical_features_to_scale:
        if col_to_scale in processed_df.columns:
            processed_df[col_to_scale] = scaler.transform(processed_df[[col_to_scale]])
    
    processed_df = processed_df[feature_columns_for_app]
    return processed_df

def make_deterministic_prediction(model, input_data, model_choice):
    """Exact copy from app.py"""
    np.random.seed(42)
    input_array = input_data.values if isinstance(input_data, pd.DataFrame) else input_data
    prediction_encoded = model.predict(input_array)
    
    # Decode predictions robustly:
    try:
        if isinstance(prediction_encoded, (list, tuple)):
            pred_arr = np.array(prediction_encoded)
        else:
            pred_arr = prediction_encoded

        if np.issubdtype(getattr(pred_arr, 'dtype', np.array(pred_arr).dtype), np.integer):
            prediction = le.inverse_transform(pred_arr)
        else:
            prediction = lb.inverse_transform(np.array([prediction_encoded]).T).flatten()
    except Exception as e:
        prediction = np.array([str(x) for x in np.atleast_1d(prediction_encoded)])
    
    return prediction

# Simulate form submission with DEFAULT/FIRST values from app.py number_input
print("="*80)
print("SIMULATING STREAMLIT FORM WITH DEFAULT VALUES")
print("="*80)
print("\nForm defaults in app.py number_input():")

num_input_fields = [
    ('Insulin Levels', 22),
    ('Age', 32),
    ('BMI', 25),
    ('Blood Pressure', 111),
    ('Cholesterol Levels', 195),
    ('Waist Circumference', 35),
    ('Blood Glucose Levels', 161),
    ('Weight Gain During Pregnancy', 15),
    ('Pancreatic Health', 48),
    ('Pulmonary Function', 70),
    ('Neurological Assessments', 2),
    ('Digestive Enzyme Levels', 46),
    ('Birth Weight', 3000),
]

input_data = {}

# Populate numerics with default values
for label, default in num_input_fields:
    input_data[label] = float(default)
    print(f"  {label}: {default}")

# Populate categoricals with FIRST option (app.py selectbox default)
print("\nForm defaults in app.py selectbox() [FIRST OPTION]:")
for col_name, possible_values in original_categorical_features_map.items():
    input_data[col_name] = possible_values[0]  # First option is default
    print(f"  {col_name}: {possible_values[0]}")

input_df = pd.DataFrame([input_data])
processed_input_df = preprocess_input(input_df)

print("\n" + "-"*80)
print("PREDICTIONS WITH DEFAULT FORM VALUES:")
print("-"*80)

models = [
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model),
]

for model_name, model in models:
    pred = make_deterministic_prediction(model, processed_input_df, model_name)
    print(f"{model_name}: {pred[0]}")

# Test 2: Change ONLY one categorical (most likely to trigger visible change)
print("\n" + "="*80)
print("TEST: CHANGE Family History from Yes → No")
print("="*80)

input_data2 = dict(input_data)
input_data2['Family History'] = 'No'  # Change to second option
input_df2 = pd.DataFrame([input_data2])
processed_input_df2 = preprocess_input(input_df2)

print("\nPREDICTIONS WITH Family History=No:")
for model_name, model in models:
    pred = make_deterministic_prediction(model, processed_input_df2, model_name)
    print(f"{model_name}: {pred[0]}")

# Test 3: Change numerical value significantly
print("\n" + "="*80)
print("TEST: CHANGE Blood Glucose Levels 161 → 300")
print("="*80)

input_data3 = dict(input_data)
input_data3['Blood Glucose Levels'] = 300
input_df3 = pd.DataFrame([input_data3])
processed_input_df3 = preprocess_input(input_df3)

print("\nPREDICTIONS WITH Blood Glucose=300:")
for model_name, model in models:
    pred = make_deterministic_prediction(model, processed_input_df3, model_name)
    print(f"{model_name}: {pred[0]}")

print("\n" + "="*80)
print("CHECKING FEATURE VECTOR CHANGES")
print("="*80)

v1 = processed_input_df.values[0]
v2 = processed_input_df2.values[0]
v3 = processed_input_df3.values[0]

diff_12 = np.sum(v1 != v2)
diff_13 = np.sum(v1 != v3)

print(f"Vector diff (default vs Family History=No): {diff_12} columns")
print(f"Vector diff (default vs Blood Glucose=300): {diff_13} columns")

if diff_12 == 0:
    print("⚠️  No change in feature vector when toggling Family History!")
if diff_13 == 0:
    print("⚠️  No change in feature vector when changing Blood Glucose!")
    
print("\nLabel Encoder index mapping:")
for i, cls in enumerate(le.classes_):
    print(f"  {i}: {cls}")
