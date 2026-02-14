"""
Debug test to understand the OHE encoding issue
"""
import pandas as pd
import numpy as np

# Feature columns exactly as in the model
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

# CURRENT INCORRECT mapping (showing all possible values)
original_categorical_features_map_incorrect = {
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

print("=" * 80)
print("DEBUGGING OHE ENCODING ISSUE")
print("=" * 80)

# Test 1: Check which OHE columns can actually be set
print("\n[TEST 1] Checking OHE column existence for different inputs")
print("-" * 80)

test_cases = [
    ('Family History', 'Yes'),
    ('Family History', 'No'),
    ('Alcohol Consumption', 'Moderate'),
    ('Alcohol Consumption', 'Heavy'),
    ('Alcohol Consumption', 'No'),
    ('Physical Activity', 'Moderate'),
    ('Physical Activity', 'High'),
    ('Dietary Habits', 'Unhealthy'),
    ('Dietary Habits', 'Healthy'),
]

for feature_name, value in test_cases:
    ohe_col_name = f"{feature_name}_{value}"
    exists = ohe_col_name in feature_columns_for_app
    status = "✓ EXISTS" if exists else "✗ MISSING"
    print(f"  {ohe_col_name:40} → {status}")

# Test 2: Create a sample dataframe with different inputs and see what gets set
print("\n\n[TEST 2] Preprocessing different inputs")
print("-" * 80)

def preprocess_input(input_df, cat_map):
    """Preprocess user input"""
    processed_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns_for_app)
    
    # Fill in numerical features
    for col in numerical_features_names:
        if col in input_df.columns:
            processed_df[col] = input_df[col].values

    # Handle categorical features
    for original_col, possible_values in cat_map.items():
        if original_col in input_df.columns:
            value = input_df[original_col].iloc[0]
            ohe_col_name = f"{original_col}_{value}"
            if ohe_col_name in processed_df.columns:
                processed_df[ohe_col_name] = 1
                print(f"  ✓ Set {ohe_col_name} = 1")
            else:
                print(f"  ✗ Column '{ohe_col_name}' not found (will leave as 0)")

    return processed_df

# Test with Family History
print("\nScenario A: Family History = 'Yes'")
data_a = {col: 50.0 for col in numerical_features_names}
for cat, vals in original_categorical_features_map_incorrect.items():
    data_a[cat] = vals[0]
data_a['Family History'] = 'Yes'

input_df_a = pd.DataFrame([data_a])
processed_df_a = preprocess_input(input_df_a, original_categorical_features_map_incorrect)

print("\nScenario B: Family History = 'No'")
data_b = dict(data_a)
data_b['Family History'] = 'No'
input_df_b = pd.DataFrame([data_b])
processed_df_b = preprocess_input(input_df_b, original_categorical_features_map_incorrect)

# Compare the Family History columns
print("\n\nComparing Family History column values:")
fh_col_idx = feature_columns_for_app.index('Family History_Yes')
print(f"  Scenario A (Yes): Family History_Yes = {processed_df_a.iloc[0, fh_col_idx]}")
print(f"  Scenario B (No):  Family History_Yes = {processed_df_b.iloc[0, fh_col_idx]}")
print(f"  → Values ARE DIFFERENT: {processed_df_a.iloc[0, fh_col_idx] != processed_df_b.iloc[0, fh_col_idx]}")

# Test 3: The real problem - Alcohol Consumption
print("\n\n[TEST 3] Problem case: Alcohol Consumption")
print("-" * 80)

print("\nScenario C: Alcohol Consumption = 'Moderate'")
data_c = {col: 50.0 for col in numerical_features_names}
for cat, vals in original_categorical_features_map_incorrect.items():
    data_c[cat] = vals[0]
data_c['Alcohol Consumption'] = 'Moderate'

input_df_c = pd.DataFrame([data_c])
processed_df_c = preprocess_input(input_df_c, original_categorical_features_map_incorrect)

print("\nScenario D: Alcohol Consumption = 'Heavy'")
data_d = dict(data_c)
data_d['Alcohol Consumption'] = 'Heavy'
input_df_d = pd.DataFrame([data_d])
processed_df_d = preprocess_input(input_df_d, original_categorical_features_map_incorrect)

# Compare the Alcohol Consumption columns
print("\n\nComparing Alcohol Consumption column values:")
ac_col_idx = feature_columns_for_app.index('Alcohol Consumption_Moderate')
print(f"  Scenario C (Moderate): Alcohol Consumption_Moderate = {processed_df_c.iloc[0, ac_col_idx]}")
print(f"  Scenario D (Heavy):    Alcohol Consumption_Moderate = {processed_df_d.iloc[0, ac_col_idx]}")
print(f"  → Values ARE DIFFERENT: {processed_df_c.iloc[0, ac_col_idx] != processed_df_d.iloc[0, ac_col_idx]}")
print(f"\n⚠️ NOTE: Both scenarios try to set the same column!")
print(f"   - 'Moderate' sets 'Alcohol_Consumption_Moderate' = 1")
print(f"   - 'Heavy' tries to set 'Alcohol_Consumption_Heavy' (doesn't exist!)")
print(f"   So both end up with the same feature values!")

EOF
