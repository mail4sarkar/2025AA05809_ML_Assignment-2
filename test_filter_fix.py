"""
Test to verify the UI filter fix - ensuring input changes produce different outputs
"""
import pandas as pd
import numpy as np
import joblib
import os

assets_dir = 'streamlit_assets'

print("=" * 80)
print("TESTING FILTER FIX - VERIFYING INPUT CHANGES AFFECT OUTPUT")
print("=" * 80)

# Load models
rf_model = joblib.load(os.path.join(assets_dir, 'rf_model.joblib'))
xgb_model = joblib.load(os.path.join(assets_dir, 'xgb_model.joblib'))
lb = joblib.load(os.path.join(assets_dir, 'label_binarizer.joblib'))
le = joblib.load(os.path.join(assets_dir, 'label_encoder.joblib'))
scaler = joblib.load(os.path.join(assets_dir, 'scaler.joblib'))

# Correct feature columns (matching updated app.py)
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
    """Preprocess user input to match model requirements"""
    processed_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns_for_app)
    
    # Fill in numerical features directly
    for col in numerical_features_names:
        if col in input_df.columns:
            processed_df[col] = input_df[col].values

    # Handle one-hot encoding for categorical features
    for original_col, possible_values in original_categorical_features_map.items():
        if original_col in input_df.columns:
            # Get the value from the input
            value = input_df[original_col].iloc[0]
            
            # Create the OHE column name and set it to 1 if the column exists
            ohe_col_name = f"{original_col}_{value}"
            if ohe_col_name in processed_df.columns:
                processed_df[ohe_col_name] = 1

    # Scale numerical features
    for col_to_scale in numerical_features_to_scale:
        if col_to_scale in processed_df.columns:
            processed_df[col_to_scale] = scaler.transform(processed_df[[col_to_scale]])
    
    # Ensure correct column order
    processed_df = processed_df[feature_columns_for_app]
    return processed_df

def make_prediction(model, input_data, model_name):
    """Make prediction with proper decoding"""
    np.random.seed(42)
    input_array = input_data.values if isinstance(input_data, pd.DataFrame) else input_data
    prediction_encoded = model.predict(input_array)
    
    if model_name == "XGBoost":
        prediction = le.inverse_transform(prediction_encoded)
    else:
        prediction = lb.inverse_transform(np.array([prediction_encoded]).T).flatten()
    
    return prediction[0]

# Test 1: Create two different patient profiles
print("\n" + "="*80)
print("TEST 1: Different numeric inputs should produce different outputs")
print("="*80)

# Patient 1: Low risk profile (low glucose, low cholesterol)
patient1_data = {col: 30.0 for col in numerical_features_names}
patient1_data['Blood Glucose Levels'] = 80.0  # Low glucose
patient1_data['Cholesterol Levels'] = 150.0   # Low cholesterol
for cat, vals in original_categorical_features_map.items():
    patient1_data[cat] = vals[0]

# Patient 2: High risk profile (high glucose, high cholesterol)
patient2_data = {col: 60.0 for col in numerical_features_names}
patient2_data['Blood Glucose Levels'] = 250.0  # High glucose
patient2_data['Cholesterol Levels'] = 300.0    # High cholesterol
for cat, vals in original_categorical_features_map.items():
    patient2_data[cat] = vals[0]

patient1_df = pd.DataFrame([patient1_data])
patient2_df = pd.DataFrame([patient2_data])

patient1_processed = preprocess_input(patient1_df)
patient2_processed = preprocess_input(patient2_df)

print("\nPatient 1 (Low Risk):")
pred1_rf = make_prediction(rf_model, patient1_processed, "RF")
pred1_xgb = make_prediction(xgb_model, patient1_processed, "XGBoost")
print(f"  Random Forest: {pred1_rf}")
print(f"  XGBoost: {pred1_xgb}")

print("\nPatient 2 (High Risk):")
pred2_rf = make_prediction(rf_model, patient2_processed, "RF")
pred2_xgb = make_prediction(xgb_model, patient2_processed, "XGBoost")
print(f"  Random Forest: {pred2_rf}")
print(f"  XGBoost: {pred2_xgb}")

if pred1_rf != pred2_rf or pred1_xgb != pred2_xgb:
    print("\n✅ SUCCESS: Numeric inputs cause output changes!")
else:
    print("\n⚠️ WARNING: Numeric inputs don't cause output changes!")

# Test 2: Different categorical inputs
print("\n" + "="*80)
print("TEST 2: Different categorical inputs should produce different outputs")
print("="*80)

patient3_data = {col: 50.0 for col in numerical_features_names}
patient4_data = {col: 50.0 for col in numerical_features_names}

# Different family history
for cat, vals in original_categorical_features_map.items():
    patient3_data[cat] = vals[0]
    patient4_data[cat] = vals[0]

patient3_data['Family History'] = 'Yes'
patient4_data['Family History'] = 'No'

patient3_df = pd.DataFrame([patient3_data])
patient4_df = pd.DataFrame([patient4_data])

patient3_processed = preprocess_input(patient3_df)
patient4_processed = preprocess_input(patient4_df)

print("\nPatient 3 (Family History: Yes):")
pred3_rf = make_prediction(rf_model, patient3_processed, "RF")
pred3_xgb = make_prediction(xgb_model, patient3_processed, "XGBoost")
print(f"  Random Forest: {pred3_rf}")
print(f"  XGBoost: {pred3_xgb}")

print("\nPatient 4 (Family History: No):")
pred4_rf = make_prediction(rf_model, patient4_processed, "RF")
pred4_xgb = make_prediction(xgb_model, patient4_processed, "XGBoost")
print(f"  Random Forest: {pred4_rf}")
print(f"  XGBoost: {pred4_xgb}")

# Check if predictions are different
if pred3_rf != pred4_rf or pred3_xgb != pred4_xgb:
    print("\n✅ SUCCESS: Categorical inputs cause output changes!")
else:
    print("\n⚠️ WARNING: Categorical inputs don't cause output changes!")

# Test 3: Verify feature shapes
print("\n" + "="*80)
print("TEST 3: Verify feature shapes and counts")
print("="*80)
print(f"\nExpected feature count: {len(feature_columns_for_app)}")
print(f"Processed data shape: {patient1_processed.shape}")
print(f"Number of columns: {patient1_processed.shape[1]}")

if patient1_processed.shape[1] == len(feature_columns_for_app):
    print("✅ SUCCESS: Feature count matches!")
else:
    print("❌ ERROR: Feature count mismatch!")

# Test 4: Check for missing values
print("\n" + "="*80)
print("TEST 4: Check for NaN or invalid values")
print("="*80)
if patient1_processed.isnull().sum().sum() > 0:
    print("❌ ERROR: Found NaN values in processed data!")
    print(patient1_processed.isnull().sum())
else:
    print("✅ SUCCESS: No NaN values!")

print("\n" + "="*80)
print("✅ FILTER FIX TEST COMPLETE")
print("="*80)
