"""
Comprehensive test to debug why all predictions are the same.
Simulates the exact preprocessing and prediction flow from app.py
"""
import pandas as pd
import numpy as np
import joblib
import os

assets_dir = 'streamlit_assets'

# Load models and encoders
print("Loading models and encoders...")
rf_model = joblib.load(os.path.join(assets_dir, 'rf_model.joblib'))
xgb_model = joblib.load(os.path.join(assets_dir, 'xgb_model.joblib'))
le = joblib.load(os.path.join(assets_dir, 'label_encoder.joblib'))
lb = joblib.load(os.path.join(assets_dir, 'label_binarizer.joblib'))
scaler = joblib.load(os.path.join(assets_dir, 'scaler.joblib'))

print(f"Label Encoder classes: {le.classes_}")
print(f"Class 0: {le.classes_[0]}")
print(f"Class 10: {le.classes_[10]}")
print()

# Feature columns from app.py
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
    'Physical Activity': ['Moderate', 'Sedentary', 'Low', 'High'],
    'Dietary Habits': ['Unhealthy', 'Healthy'],
    'Ethnicity': ['Asian', 'Black', 'Hispanic', 'Other', 'White'],
    'Socioeconomic Factors': ['High Income', 'Upper Class', 'Low Income', 'Middle Class'],
    'Smoking Status': ['Yes', 'No'],
    'Alcohol Consumption': ['Moderate', 'Heavy', 'No'],
    'Glucose Tolerance Test': ['Impaired', 'Normal'],
    'History of PCOS': ['Yes', 'No'],
    'Previous Gestational Diabetes': ['Yes', 'No'],
    'Pregnancy History': ['Complications', 'Normal'],
    'Cystic Fibrosis Diagnosis': ['Yes', 'No'],
    'Steroid Use History': ['Yes', 'No'],
    'Genetic Testing': ['Positive', 'Negative'],
    'Liver Function Tests': ['Abnormal', 'Normal'],
    'Urine Test': ['Ketones Present', 'Protein Present', 'Glucose Present', 'Normal'],
    'Early Onset Symptoms': ['Yes', 'No']
}

def preprocess_input(input_df):
    """Exact copy of preprocessing from app.py"""
    processed_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns_for_app)
    
    # Fill in numerical features directly
    for col in numerical_features_names:
        if col in input_df.columns:
            processed_df[col] = input_df[col].values

    # Handle one-hot encoding for categorical features
    for original_col, possible_values in original_categorical_features_map.items():
        if original_col in input_df.columns:
            value = input_df[original_col].iloc[0]
            ohe_col_name = f"{original_col}_{value}"
            if ohe_col_name in processed_df.columns:
                processed_df[ohe_col_name] = 1

    # Scale numerical features
    for col_to_scale in numerical_features_to_scale:
        if col_to_scale in processed_df.columns:
            processed_df[col_to_scale] = scaler.transform(processed_df[[col_to_scale]])
    
    processed_df = processed_df[feature_columns_for_app]
    return processed_df

def decode_prediction(predicted_encoded, model_choice):
    """Decode prediction - matches app.py logic"""
    try:
        if isinstance(predicted_encoded, (list, tuple)):
            pred_arr = np.array(predicted_encoded)
        else:
            pred_arr = predicted_encoded

        if np.issubdtype(getattr(pred_arr, 'dtype', np.array(pred_arr).dtype), np.integer):
            prediction = le.inverse_transform(pred_arr)
        else:
            prediction = lb.inverse_transform(np.array([predicted_encoded]).T).flatten()
    except Exception as e:
        prediction = np.array([str(x) for x in np.atleast_1d(predicted_encoded)])
    
    return prediction[0]

# Test 1: Create different patient profiles
print("="*80)
print("TEST 1: Different numeric inputs")
print("="*80)

def create_patient(glucose, cholesterol):
    patient_data = {col: 50.0 for col in numerical_features_names}
    patient_data['Blood Glucose Levels'] = glucose
    patient_data['Cholesterol Levels'] = cholesterol
    
    for cat, vals in original_categorical_features_map.items():
        patient_data[cat] = vals[0]  # Pick first option
    
    return patient_data

# Patient 1: Low glucose
patient1 = create_patient(glucose=90, cholesterol=150)
p1_df = pd.DataFrame([patient1])
p1_processed = preprocess_input(p1_df)

# Patient 2: High glucose
patient2 = create_patient(glucose=300, cholesterol=350)
p2_df = pd.DataFrame([patient2])
p2_processed = preprocess_input(p2_df)

print("\nPatient 1 (Low glucose=90, cholesterol=150)")
p1_rf_enc = rf_model.predict(p1_processed.values)[0]
p1_rf_dec = decode_prediction(np.array([p1_rf_enc]), "RF")
print(f"  RF: encoded={p1_rf_enc}, decoded={p1_rf_dec}")

p1_xgb_enc = xgb_model.predict(p1_processed.values)[0]
p1_xgb_dec = decode_prediction(np.array([p1_xgb_enc]), "XGBoost")
print(f"  XGB: encoded={p1_xgb_enc}, decoded={p1_xgb_dec}")

print("\nPatient 2 (High glucose=300, cholesterol=350)")
p2_rf_enc = rf_model.predict(p2_processed.values)[0]
p2_rf_dec = decode_prediction(np.array([p2_rf_enc]), "RF")
print(f"  RF: encoded={p2_rf_enc}, decoded={p2_rf_dec}")

p2_xgb_enc = xgb_model.predict(p2_processed.values)[0]
p2_xgb_dec = decode_prediction(np.array([p2_xgb_enc]), "XGBoost")
print(f"  XGB: encoded={p2_xgb_enc}, decoded={p2_xgb_dec}")

print()
if p1_rf_enc == p2_rf_enc:
    print("⚠️  RF predictions are IDENTICAL for different glucose levels")
    # Check if features actually differ
    diff_count = np.sum(p1_processed.values != p2_processed.values)
    print(f"   Feature vector differences: {diff_count} out of {len(feature_columns_for_app)}")
    if diff_count > 0:
        idx_diff = np.where(p1_processed.values != p2_processed.values)[1]
        print(f"   Different columns: {[feature_columns_for_app[i] for i in idx_diff]}")
else:
    print("✅ RF predictions DIFFER for different glucose levels")

if p1_xgb_enc == p2_xgb_enc:
    print("⚠️  XGB predictions are IDENTICAL")
else:
    print("✅ XGB predictions DIFFER")

# Test 2: Different categorical combinations
print("\n" + "="*80)
print("TEST 2: Low-risk vs High-risk categorical profiles")
print("="*80)

def create_low_risk_patient():
    patient = create_patient(glucose=100, cholesterol=180)
    patient['Family History'] = 'No'
    patient['Smoking Status'] = 'No'
    return patient

def create_high_risk_patient():
    patient = create_patient(glucose=100, cholesterol=180)
    patient['Family History'] = 'Yes'
    patient['Smoking Status'] = 'Yes'
    return patient

p_low = create_low_risk_patient()
p_low_df = pd.DataFrame([p_low])
p_low_processed = preprocess_input(p_low_df)

p_high = create_high_risk_patient()
p_high_df = pd.DataFrame([p_high])
p_high_processed = preprocess_input(p_high_df)

print("\nLow-risk patient")
p_low_rf_enc = rf_model.predict(p_low_processed.values)[0]
p_low_rf_dec = decode_prediction(np.array([p_low_rf_enc]), "RF")
print(f"  RF: {p_low_rf_dec}")

print("\nHigh-risk patient")
p_high_rf_enc = rf_model.predict(p_high_processed.values)[0]
p_high_rf_dec = decode_prediction(np.array([p_high_rf_enc]), "RF")
print(f"  RF: {p_high_rf_dec}")

if p_low_rf_enc == p_high_rf_enc:
    print("\n⚠️  Predictions are IDENTICAL despite different risk profiles")
    diff_count = np.sum(p_low_processed.values != p_high_processed.values)
    print(f"   Feature differences: {diff_count}")
else:
    print("\n✅ Predictions DIFFER for different risk profiles")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf all predictions are the same despite different inputs:")
print("1. The models might have been trained on features in a different order")
print("2. The preprocessing might be creating identical vectors")
print("3. The encoded predictions from the model might be wrong")
print("\nCheck the feature differences count above - if it's 0, preprocessing isn't changing vectors!")
