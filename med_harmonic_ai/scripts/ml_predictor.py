"""
Machine learning predictor for the Medicinal Harmonic Resonance AI project.
This module trains a model to predict the efficacy of compounds for diabetes treatment.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional

# Set proper path for Python 3.9 environment
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

# Add data processor script directory to path
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__))
sys.path.append(SCRIPTS_DIR)

try:
    from data_processor import MedicinalDataProcessor
except ImportError:
    print("Could not import MedicinalDataProcessor. Make sure data_processor.py is in the same directory.")

class HarmonicResonancePredictor:
    """Predictor for compound efficacy based on medicinal harmonic resonance."""
    
    def __init__(self, model_dir: str = "../models"):
        """Initialize the predictor with model directory."""
        # Convert relative path to absolute if needed
        if not os.path.isabs(model_dir):
            self.model_dir = os.path.abspath(os.path.join(BASE_DIR, model_dir))
        else:
            self.model_dir = model_dir
            
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Also ensure app-specific model directory exists
        self.app_model_dir = os.path.join(BASE_DIR, 'med_harmonic_ai', 'models')
        os.makedirs(self.app_model_dir, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        print(f"Model will be saved to {self.model_dir} and {self.app_model_dir}")
        
    def load_and_prepare_data(self):
        """Load data and prepare it for training."""
        # Use the data processor to prepare the dataset
        processor = MedicinalDataProcessor(os.path.join(BASE_DIR, 'data'))
        
        # Prepare the processed dataset for diabetes
        processed_df = processor.prepare_diabetes_dataset()
        
        # Add harmonic features
        processed_df = processor.add_harmonic_features(processed_df)
        
        # Save the processed dataset
        os.makedirs(os.path.join(BASE_DIR, 'data/processed'), exist_ok=True)
        processed_path = os.path.join(BASE_DIR, 'data/processed/diabetes_compounds.csv')
        processed_df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path}")
        
        return processed_df
    
    def prepare_features(self, df):
        """Prepare features and target for model training."""
        # Define target - whether compound is effective for diabetes
        target_col = 'effective_for_diabetes'
        
        # Store feature names for later use
        self.feature_names = [col for col in df.columns if col != target_col and col != 'name' and col != 'system']
        
        # Split into features and target
        X = df[self.feature_names]
        y = df[target_col]
        
        return X, y
    
    def train_model(self, X, y):
        """Train a Random Forest model for prediction."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train a Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Model performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        self.model = model
        return model
    
    def save_model(self, filename="harmonic_diabetes_predictor.joblib"):
        """Save the trained model to a file using both absolute and relative paths."""
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")
        
        # Save with relative path
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        # Save to main models directory
        main_model_path = os.path.join(self.model_dir, filename)
        joblib.dump(model_data, main_model_path)
        print(f"Model saved to {main_model_path}")
        
        # Also save to app-specific model directory for easier loading
        app_model_path = os.path.join(self.app_model_dir, filename)
        joblib.dump(model_data, app_model_path)
        print(f"Model also saved to {app_model_path}")
        
        # Save feature importance plot
        self.plot_feature_importance()
        
    def plot_feature_importance(self, feature_names=None):
        """Plot and save the feature importance from the trained model."""
        if self.model is None:
            raise ValueError("No model available! Train the model first.")
        
        if feature_names is None:
            feature_names = self.feature_names
            
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance for Diabetes Efficacy')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Save to main models directory
        main_plot_path = os.path.join(self.model_dir, 'feature_importance.png')
        plt.savefig(main_plot_path)
        
        # Also save to app models directory
        app_plot_path = os.path.join(self.app_model_dir, 'feature_importance.png')
        plt.savefig(app_plot_path)
        
        plt.close()
        print(f"Feature importance plot saved to {main_plot_path} and {app_plot_path}")
        
    def load_model(self, filename="harmonic_diabetes_predictor.joblib"):
        """Load a trained model from a file."""
        try:
            # Try loading from main models directory first
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                print(f"Model loaded from {model_path}")
                return True
                
            # Try app-specific model directory next
            app_model_path = os.path.join(self.app_model_dir, filename)
            if os.path.exists(app_model_path):
                model_data = joblib.load(app_model_path)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                print(f"Model loaded from {app_model_path}")
                return True
                
            # If not found in either location
            print(f"Model file not found at {model_path} or {app_model_path}")
            return False
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_single_compound(self, compound_data):
        """
        Predict the efficacy of a single compound.
        
        Args:
            compound_data (dict): Dictionary of compound properties
            
        Returns:
            dict: Prediction results including efficacy and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded! Load or train the model first.")
        
        if not self.feature_names:
            raise ValueError("Feature names not available. Load or train the model first.")
        
        # Convert the compound data to a DataFrame with the correct features
        compound_df = pd.DataFrame([compound_data])
        
        # Ensure all needed features are present
        for feature in self.feature_names:
            if feature not in compound_df.columns:
                compound_df[feature] = 0  # Default value for missing features
        
        # Select only the features the model was trained on
        X = compound_df[self.feature_names]
        
        # Predict efficacy
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0][1]  # Probability of class 1
        
        return {
            'is_effective': bool(prediction),
            'confidence': float(confidence)
        }
    
    def get_feature_importance(self):
        """Get feature importance rankings."""
        if self.model is None:
            raise ValueError("No model available! Train the model first.")
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return {self.feature_names[i]: importances[i] for i in indices}

if __name__ == "__main__":
    # Create and train the predictor
    predictor = HarmonicResonancePredictor()
    
    # Load and prepare the data
    data = predictor.load_and_prepare_data()
    
    # Prepare features
    X, y = predictor.prepare_features(data)
    
    # Train the model
    predictor.train_model(X, y)
    
    # Save the model
    predictor.save_model()
    
    # Test a new compound prediction
    test_compound = {
        'is_bitter': 1,
        'is_cooling': 0,
        'reduces_kapha': 1,
        'active_compounds_count': 4,
        'has_blood_sugar_effect': 1,
        'is_hot': 0,
        'reduces_phlegm': 1,
        'half_life_hours': 5.0,
        'bioavailability': 0.6,
        'has_thirst_symptom': 1,
        'has_urination_symptom': 1,
        'system_frequency': 0.5,
        'efficacy_amplitude': 0.8,
        'phase_alignment': 0.7,
        'resonance_potential': 0.75
    }
    
    prediction = predictor.predict_single_compound(test_compound)
    print(f"\nExample prediction for a new compound:")
    print(f"Effective for diabetes: {prediction['is_effective']}")
    print(f"Confidence: {prediction['confidence']:.2f}") 