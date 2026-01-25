"""
Test script to debug XGBoost model specifically
"""
import pandas as pd
import numpy as np
import joblib
import os

# Load assets
assets_dir = 'streamlit_assets'

print("=" * 80)
print("TESTING XGBOOST MODEL")
print("=" * 80)

try:
    print("\n[1] Loading XGBoost model...")
    xgb_model = joblib.load(os.path.join(assets_dir, 'xgb_model.joblib'))
    print(f"✓ XGBoost model loaded successfully")
    print(f"  Model type: {type(xgb_model)}")
    print(f"  Expected features: {xgb_model.n_features_in_}")
except Exception as e:
    print(f"✗ Error loading XGBoost: {e}")
    exit(1)

try:
    print("\n[2] Loading label encoder...")
    le = joblib.load(os.path.join(assets_dir, 'label_encoder.joblib'))
    print(f"✓ Label encoder loaded successfully")
    print(f"  Encoder type: {type(le)}")
    print(f"  Classes: {le.classes_}")
    print(f"  Number of classes: {len(le.classes_)}")
except Exception as e:
    print(f"✗ Error loading label encoder: {e}")
    exit(1)

try:
    print("\n[3] Testing XGBoost prediction with random data...")
    # Create sample data with 38 features (matching model expectations)
    sample_data = np.random.randn(1, 38)
    print(f"  Sample input shape: {sample_data.shape}")
    
    # Make prediction
    prediction_encoded = xgb_model.predict(sample_data)
    print(f"  Raw prediction: {prediction_encoded}")
    print(f"  Raw prediction type: {type(prediction_encoded)}")
    
    # Decode prediction
    prediction_decoded = le.inverse_transform(prediction_encoded)
    print(f"  Decoded prediction: {prediction_decoded}")
    print(f"  Decoded prediction type: {type(prediction_decoded)}")
    print(f"✓ XGBoost prediction successful!")
    
except Exception as e:
    print(f"✗ Error in prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    print("\n[4] Testing prediction probabilities...")
    proba = xgb_model.predict_proba(sample_data)
    print(f"  Prediction probabilities shape: {proba.shape}")
    print(f"  Max probability: {proba.max():.4f}")
    print(f"  Prediction confidence: {proba.max():.2%}")
    print(f"✓ Probabilities computed successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not compute probabilities: {e}")

print("\n" + "=" * 80)
print("✅ XGBOOST MODEL IS WORKING CORRECTLY!")
print("=" * 80)
