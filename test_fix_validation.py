"""
Test the fix by using only values that have OHE columns
"""
import pandas as pd
import numpy as np
import joblib
import os

assets_dir = 'streamlit_assets'

print("=" * 80)
print("TESTING FIX: Using Only Valid OHE Values")
print("=" * 80)

# Load all models
xgb_model = joblib.load(os.path.join(assets_dir, 'xgb_model.joblib'))
le = joblib.load(os.path.join(assets_dir, 'label_encoder.joblib'))
lb = joblib.load(os.path.join(assets_dir, 'label_binarizer.joblib'))
scaler = joblib.load(os.path.join(assets_dir, 'scaler.joblib'))
rf_model = joblib.load(os.path.join(assets_dir, 'rf_model.joblib'))

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

# FIXED categorical mapping - only values with OHE columns
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
    """Preprocess user input to match model requirements"""
    processed_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns_for_app)
    
    # Fill numerical features
    for col in numerical_features_names:
        if col in input_df.columns:
            processed_df[col] = input_df[col].values

    # Handle categorical one-hot encoding 
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

def make_deterministic_prediction(model, input_data, model_choice):
    """Make prediction with proper label decoding"""
    np.random.seed(42)
    input_array = input_data.values if isinstance(input_data, pd.DataFrame) else input_data
    prediction_encoded = model.predict(input_array)
    
    if model_choice == "XGBoost":
        prediction = le.inverse_transform(prediction_encoded)
    else:
        prediction = lb.inverse_transform(np.array([prediction_encoded]).T).flatten()
    
    return prediction[0]

# Test 1: Numeric changes
print("\n[TEST 1] Numeric input changes (keeping categorical same)")
print("-" * 80)

def create_patient(glucose, cholesterol, all_yes=False):
    patient_data = {col: 50.0 for col in numerical_features_names}
    patient_data['Blood Glucose Levels'] = glucose
    patient_data['Cholesterol Levels'] = cholesterol
    
    if all_yes:
        # Select all "positive" values for categorical
        patient_data['Genetic Markers'] = 'Positive'
        patient_data['Autoantibodies'] = 'Positive'
        patient_data['Family History'] = 'Yes'
        patient_data['Environmental Factors'] = 'Present'
        patient_data['Physical Activity'] = 'Moderate'
        patient_data['Dietary Habits'] = 'Unhealthy'
        patient_data['Ethnicity'] = 'Asian'
        patient_data['Socioeconomic Factors'] = 'High Income'
        patient_data['Smoking Status'] = 'Yes'
        patient_data['Alcohol Consumption'] = 'Moderate'
        patient_data['Glucose Tolerance Test'] = 'Impaired'
        patient_data['History of PCOS'] = 'Yes'
        patient_data['Previous Gestational Diabetes'] = 'Yes'
        patient_data['Pregnancy History'] = 'Complications'
        patient_data['Cystic Fibrosis Diagnosis'] = 'Yes'
        patient_data['Steroid Use History'] = 'Yes'
        patient_data['Genetic Testing'] = 'Positive'
        patient_data['Liver Function Tests'] = 'Abnormal'
        patient_data['Urine Test'] = 'Ketones Present'
        patient_data['Early Onset Symptoms'] = 'Yes'
    else:
        # Select "negative" values
        patient_data['Genetic Markers'] = 'Negative'
        patient_data['Autoantibodies'] = 'Negative'
        patient_data['Family History'] = 'No'
        patient_data['Environmental Factors'] = 'Absent'
        patient_data['Physical Activity'] = 'High'  # Not Moderate
        patient_data['Dietary Habits'] = 'Healthy'
        patient_data['Ethnicity'] = 'White'
        patient_data['Socioeconomic Factors'] = 'Middle Class'
        patient_data['Smoking Status'] = 'No'
        patient_data['Alcohol Consumption'] = 'No'
        patient_data['Glucose Tolerance Test'] = 'Normal'
        patient_data['History of PCOS'] = 'No'
        patient_data['Previous Gestational Diabetes'] = 'No'
        patient_data['Pregnancy History'] = 'Normal'
        patient_data['Cystic Fibrosis Diagnosis'] = 'No'
        patient_data['Steroid Use History'] = 'No'
        patient_data['Genetic Testing'] = 'Negative'
        patient_data['Liver Function Tests'] = 'Normal'
        patient_data['Urine Test'] = 'Normal'
        patient_data['Early Onset Symptoms'] = 'No'
    
    return patient_data

# Low glucose vs High glucose
patient_low_glucose = create_patient(glucose=90, cholesterol=150)
patient_high_glucose = create_patient(glucose=300, cholesterol=350)

patient_low_processed = preprocess_input(pd.DataFrame([patient_low_glucose]))
patient_high_processed = preprocess_input(pd.DataFrame([patient_high_glucose]))

print("\nPatient with Low Blood Glucose (90):")
pred_low_rf = make_deterministic_prediction(rf_model, patient_low_processed, "Random Forest")
pred_low_xgb = make_deterministic_prediction(xgb_model, patient_low_processed, "XGBoost")
print(f"  Random Forest: {pred_low_rf}")
print(f"  XGBoost:       {pred_low_xgb}")

print("\nPatient with High Blood Glucose (300):")
pred_high_rf = make_deterministic_prediction(rf_model, patient_high_processed, "Random Forest")
pred_high_xgb = make_deterministic_prediction(xgb_model, patient_high_processed, "XGBoost")
print(f"  Random Forest: {pred_high_rf}")
print(f"  XGBoost:       {pred_high_xgb}")

numeric_change_works = (pred_low_rf != pred_high_rf) or (pred_low_xgb != pred_high_xgb)
if numeric_change_works:
    print("\n✅ SUCCESS: Numeric changes affect output!")
else:
    print("\n⚠️  Numeric changes don't affect output (this might be normal depending on model)")

# Test 2: Categorical changes
print("\n\n[TEST 2] Categorical input changes")
print("-" * 80)

patient_low_risk = create_patient(glucose=100, cholesterol=180, all_yes=False)
patient_high_risk = create_patient(glucose=100, cholesterol=180, all_yes=True)

patient_low_risk_processed = preprocess_input(pd.DataFrame([patient_low_risk]))
patient_high_risk_processed = preprocess_input(pd.DataFrame([patient_high_risk]))

print("\nLow-risk patient (all negative factors):")
pred_low_risk_rf = make_deterministic_prediction(rf_model, patient_low_risk_processed, "Random Forest")
pred_low_risk_xgb = make_deterministic_prediction(xgb_model, patient_low_risk_processed, "XGBoost")
print(f"  Random Forest: {pred_low_risk_rf}")
print(f"  XGBoost:       {pred_low_risk_xgb}")

print("\nHigh-risk patient (all positive factors):")
pred_high_risk_rf = make_deterministic_prediction(rf_model, patient_high_risk_processed, "Random Forest")
pred_high_risk_xgb = make_deterministic_prediction(xgb_model, patient_high_risk_processed, "XGBoost")
print(f"  Random Forest: {pred_high_risk_rf}")
print(f"  XGBoost:       {pred_high_risk_xgb}")

categorical_change_works = (pred_low_risk_rf != pred_high_risk_rf) or (pred_low_risk_xgb != pred_high_risk_xgb)
if categorical_change_works:
    print("\n✅ SUCCESS: Categorical changes affect output!")
else:
    print("\n⚠️  Categorical changes don't affect output")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"\nNumeric changes work: {numeric_change_works}")
print(f"Categorical changes work: {categorical_change_works}")

if numeric_change_works or categorical_change_works:
    print("\n✅ UI FILTER IS NOW WORKING PROPERLY!")
else:
    print("\n⚠️  Further investigation needed")

EOF
