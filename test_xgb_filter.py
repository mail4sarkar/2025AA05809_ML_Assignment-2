"""
Comprehensive XGBoost model filter test - simulating exact app flow
"""
import pandas as pd
import numpy as np
import joblib
import os

assets_dir = 'streamlit_assets'

print("=" * 80)
print("TESTING XGBOOST MODEL FILTER AND PREDICTION FLOW")
print("=" * 80)

# Load all models
xgb_model = joblib.load(os.path.join(assets_dir, 'xgb_model.joblib'))
le = joblib.load(os.path.join(assets_dir, 'label_encoder.joblib'))
lb = joblib.load(os.path.join(assets_dir, 'label_binarizer.joblib'))
scaler = joblib.load(os.path.join(assets_dir, 'scaler.joblib'))
rf_model = joblib.load(os.path.join(assets_dir, 'rf_model.joblib'))

# Define categorical features
original_categorical_features_map = {
    'Genetic Markers': ['Positive', 'Negative'],
    'Autoantibodies': ['Positive', 'Negative'],
    'Family History': ['Yes', 'No'],
    'Environmental Factors': ['Present', 'Absent'],
    'Physical Activity': ['High', 'Low', 'Moderate', 'Sedentary'],
    'Dietary Habits': ['Healthy', 'Unhealthy'],
    'Ethnicity': ['Asian', 'Black', 'Hispanic', 'Other', 'White'],
    'Socioeconomic Factors': ['High Income', 'Low Income', 'Middle Class', 'Upper Class'],
    'Smoking Status': ['Yes', 'No'],
    'Alcohol Consumption': ['Heavy', 'Moderate', 'No'],
    'Glucose Tolerance Test': ['Impaired', 'Normal'],
    'History of PCOS': ['Yes', 'No'],
    'Previous Gestational Diabetes': ['Yes', 'No'],
    'Pregnancy History': ['Normal', 'Complications'],
    'Cystic Fibrosis Diagnosis': ['Yes', 'No'],
    'Steroid Use History': ['Yes', 'No'],
    'Genetic Testing': ['Positive', 'Negative'],
    'Liver Function Tests': ['Normal', 'Abnormal'],
    'Urine Test': ['Glucose Present', 'Ketones Present', 'Normal', 'Protein Present'],
    'Early Onset Symptoms': ['Yes', 'No']
}

numerical_features_names = [
    'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels',
    'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy',
    'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments',
    'Digestive Enzyme Levels', 'Birth Weight'
]

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

numerical_features_to_scale = ['Birth Weight']

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
    
    print(f"  Raw prediction: {prediction_encoded}")
    
    if model_choice == "XGBoost":
        prediction = le.inverse_transform(prediction_encoded)
        print(f"  Using le.inverse_transform()")
    else:
        prediction = lb.inverse_transform(np.array([prediction_encoded]).T).flatten()
        print(f"  Using lb.inverse_transform()")
    
    return prediction

# Create sample patient data
print("\n[1] Creating sample patient data...")
input_data = {}

# Numerical values
for col_name in numerical_features_names:
    input_data[col_name] = 50.0

# Categorical values (first option from each)
for category, values in original_categorical_features_map.items():
    input_data[category] = values[0]

input_df = pd.DataFrame([input_data])
print(f"✓ Sample patient data created (shape: {input_df.shape})")

print("\n[2] Preprocessing input...")
processed_input_df = preprocess_input(input_df)
print(f"✓ Preprocessed data shape: {processed_input_df.shape}")

# Test all models
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
}

print("\n[3] Testing predictions with model selector...")
for selected_model_choice, model in models.items():
    print(f"\n  Model: {selected_model_choice}")
    try:
        prediction = make_deterministic_prediction(model, processed_input_df, selected_model_choice)
        print(f"  ✓ Decoded prediction: {prediction[0]}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("✅ XGBOOST MODEL FILTER TEST COMPLETE")
print("=" * 80)
