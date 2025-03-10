"""
Launcher for the Medicinal Harmonic Resonance AI application on Streamlit Cloud.
This file imports and runs the main application.
"""
import os
import sys

# Add the necessary paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
API_DIR = os.path.join(ROOT_DIR, 'med_harmonic_ai', 'api')
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'med_harmonic_ai', 'scripts')

# Add directories to path
sys.path.append(ROOT_DIR)
sys.path.append(API_DIR)
sys.path.append(SCRIPTS_DIR)

# Create necessary directories if they don't exist
os.makedirs(os.path.join(ROOT_DIR, 'med_harmonic_ai', 'models'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'med_harmonic_ai', 'data', 'raw'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'med_harmonic_ai', 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'data', 'raw'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)

# Run data collection and model training if needed
try:
    # Check if model exists
    model_path = os.path.join(ROOT_DIR, 'med_harmonic_ai', 'models', 'harmonic_diabetes_predictor.joblib')
    if not os.path.exists(model_path):
        print("Model not found. Running data collection and model training...")
        # Import and run data collection
        from med_harmonic_ai.scripts.data_collectors import collect_all_data
        print("Collecting data...")
        collect_all_data()
        
        # Import and run model training
        from med_harmonic_ai.scripts.ml_predictor import HarmonicResonancePredictor
        print("Training model...")
        predictor = HarmonicResonancePredictor(model_dir=os.path.join(ROOT_DIR, 'med_harmonic_ai', 'models'))
        data = predictor.load_and_prepare_data()
        X, y = predictor.prepare_features(data)
        predictor.train_model(X, y)
        predictor.save_model()
        print("Model trained and saved successfully.")
    else:
        print(f"Model found at {model_path}")
except Exception as e:
    print(f"Error preparing model: {e}")
    print("Continuing with app startup...")

# Import the main app
from med_harmonic_ai.api.app import main

# Run the app
if __name__ == "__main__":
    main() 