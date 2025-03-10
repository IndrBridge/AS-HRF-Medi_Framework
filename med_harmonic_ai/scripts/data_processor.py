"""
Data processor for the Medicinal Harmonic Resonance AI project.
This module processes raw compound data for machine learning.
"""
import os
import sys
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Any, Optional

# Set proper path for Python 3.9 environment
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

class MedicinalDataProcessor:
    """Process raw data from various medical systems for machine learning."""
    
    def __init__(self, data_dir: str = "../data"):
        """Initialize with data directory."""
        # Convert relative path to absolute if needed
        if not os.path.isabs(data_dir):
            self.data_dir = os.path.abspath(os.path.join(BASE_DIR, data_dir))
        else:
            self.data_dir = data_dir
            
        # Create processed data directory
        self.processed_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        print(f"Data directory: {self.data_dir}")
        print(f"Processed directory: {self.processed_dir}")
    
    def load_raw_data(self, system: str) -> pd.DataFrame:
        """
        Load raw data for a specific medical system.
        
        Args:
            system (str): Medical system name (ayurvedic, allopathic, unani, homeopathic)
            
        Returns:
            pd.DataFrame: DataFrame containing raw data
        """
        # Define possible file paths (for flexibility)
        possible_paths = [
            os.path.join(self.data_dir, "raw", f"{system.lower()}_compounds.csv"),
            os.path.join(BASE_DIR, "med_harmonic_ai", "data", "raw", f"{system.lower()}_compounds.csv")
        ]
        
        # Try each path until we find a file
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading {system} data from {path}")
                return pd.read_csv(path)
                
        # If we can't find the file, raise an error
        raise FileNotFoundError(f"No data file found for {system} system. Tried paths: {possible_paths}")
    
    def _extract_common_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and standardize common features across all medical systems.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw data
            
        Returns:
            pd.DataFrame: DataFrame with standardized features
        """
        # Ensure there are no duplicate compounds
        df = df.drop_duplicates(subset=['name'])
        
        # Rename columns for consistency
        if 'effective_for_diabetes' in df.columns:
            df = df.rename(columns={'effective_for_diabetes': 'effective_for_diabetes'})
        
        # Set numeric columns as float
        numeric_cols = [
            'efficacy_amplitude', 'system_frequency', 
            'phase_alignment', 'resonance_potential',
            'half_life_hours', 'bioavailability'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Set binary columns as int
        binary_cols = [
            'is_bitter', 'is_cooling', 'reduces_kapha',
            'is_hot', 'reduces_phlegm', 'has_thirst_symptom',
            'has_urination_symptom', 'has_blood_sugar_effect',
            'effective_for_diabetes'
        ]
        
        for col in binary_cols:
            if col in df.columns:
                # Convert True/False to 1/0 if needed
                if df[col].dtype == bool:
                    df[col] = df[col].astype(int)
        
        return df
    
    def prepare_diabetes_dataset(self) -> pd.DataFrame:
        """
        Prepare a dataset specifically for diabetes treatment prediction.
        
        Returns:
            pd.DataFrame: Processed dataset ready for machine learning
        """
        # Try to load the combined data first
        try:
            combined_path = os.path.join(self.data_dir, "raw", "all_compounds.csv")
            df = pd.read_csv(combined_path)
            print(f"Loaded combined data from {combined_path}")
        except:
            # If combined data not available, load and combine individual systems
            try:
                systems = ["ayurvedic", "allopathic", "unani", "homeopathic"]
                dfs = []
                
                for system in systems:
                    try:
                        system_df = self.load_raw_data(system)
                        dfs.append(system_df)
                    except FileNotFoundError as e:
                        print(f"Warning: {e}")
                
                if not dfs:
                    raise FileNotFoundError("No data files found for any medical system")
                    
                df = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                print(f"Error loading data: {e}")
                # Create minimal synthetic data for testing
                df = self._create_synthetic_data()
        
        # Process features
        processed_df = self._extract_common_features(df)
        
        # Fill missing values
        processed_df = self._fill_missing_values(processed_df)
        
        # Save processed data
        processed_path = os.path.join(self.processed_dir, "diabetes_compounds.csv")
        processed_df.to_csv(processed_path, index=False)
        print(f"Processed diabetes dataset saved to {processed_path}")
        
        return processed_df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the dataset with appropriate defaults."""
        # Create a copy to avoid modifying the original
        filled_df = df.copy()
        
        # Fill missing numeric values with system-specific defaults or medians
        for col in filled_df.select_dtypes(include=['float64', 'int64']).columns:
            if filled_df[col].isna().any():
                # Use median by system if possible, otherwise overall median
                filled_df[col] = filled_df.groupby('system')[col].transform(
                    lambda x: x.fillna(x.median() if not x.median() != x.median() else 0)
                )
                # Fill any remaining NaNs with overall median or 0
                filled_df[col] = filled_df[col].fillna(filled_df[col].median() if filled_df[col].median() == filled_df[col].median() else 0)
        
        # Fill categorical columns with mode
        for col in filled_df.select_dtypes(include=['object']).columns:
            if filled_df[col].isna().any():
                filled_df[col] = filled_df[col].fillna(filled_df[col].mode()[0])
        
        return filled_df
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for testing when no real data is available."""
        print("WARNING: Creating synthetic data for testing purposes only")
        
        systems = ["Ayurvedic", "Allopathic", "Unani", "Homeopathic"]
        compounds = []
        
        for system in systems:
            for i in range(5):  # 5 compounds per system
                compound = {
                    "name": f"{system}_Compound_{i+1}",
                    "system": system,
                    "effective_for_diabetes": random.choice([True, False]),
                    "is_bitter": random.choice([0, 1]),
                    "is_cooling": random.choice([0, 1]),
                    "reduces_kapha": random.choice([0, 1]),
                    "is_hot": random.choice([0, 1]),
                    "reduces_phlegm": random.choice([0, 1]),
                    "has_thirst_symptom": random.choice([0, 1]),
                    "has_urination_symptom": random.choice([0, 1]),
                    "active_compounds_count": random.randint(1, 10),
                    "has_blood_sugar_effect": random.choice([0, 1]),
                    "half_life_hours": random.uniform(0.5, 24.0),
                    "bioavailability": random.uniform(0.1, 1.0),
                    "system_frequency": random.uniform(0.2, 0.8),
                    "efficacy_amplitude": random.uniform(0.3, 0.9),
                    "phase_alignment": random.uniform(0.3, 0.9),
                    "resonance_potential": random.uniform(0.3, 0.9)
                }
                compounds.append(compound)
        
        return pd.DataFrame(compounds)
    
    def add_harmonic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add harmonic resonance features to the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional harmonic features
        """
        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # System frequency mapping if not already present
        if 'system_frequency' not in enhanced_df.columns:
            system_freq_map = {
                'Ayurvedic': 0.33,
                'Allopathic': 0.67,
                'Unani': 0.50,
                'Homeopathic': 0.25
            }
            
            # Apply mapping
            if 'system' in enhanced_df.columns:
                enhanced_df['system_frequency'] = enhanced_df['system'].map(
                    lambda x: system_freq_map.get(x, 0.5)
                )
        
        # Efficacy amplitude if not present (based on effectiveness)
        if 'efficacy_amplitude' not in enhanced_df.columns:
            if 'effective_for_diabetes' in enhanced_df.columns:
                enhanced_df['efficacy_amplitude'] = enhanced_df['effective_for_diabetes'].map(
                    lambda x: random.uniform(0.7, 0.95) if x else random.uniform(0.2, 0.6)
                )
        
        # Phase alignment if not present
        if 'phase_alignment' not in enhanced_df.columns:
            enhanced_df['phase_alignment'] = enhanced_df.apply(
                lambda row: self._calculate_phase_alignment(row),
                axis=1
            )
        
        # Resonance potential if not present
        if 'resonance_potential' not in enhanced_df.columns:
            enhanced_df['resonance_potential'] = enhanced_df.apply(
                lambda row: self._calculate_resonance_potential(row),
                axis=1
            )
        
        return enhanced_df
    
    def _calculate_phase_alignment(self, row: pd.Series) -> float:
        """Calculate phase alignment based on compound properties."""
        base_alignment = 0.5
        
        # Add contributions from various properties
        if 'has_blood_sugar_effect' in row and row['has_blood_sugar_effect']:
            base_alignment += 0.2
            
        if 'active_compounds_count' in row:
            # More active compounds can lead to more possibilities for alignment
            base_alignment += min(0.1, row['active_compounds_count'] * 0.01)
            
        # Add some randomness
        alignment = base_alignment + random.uniform(-0.1, 0.1)
        
        # Ensure within range [0, 1]
        return max(0.0, min(1.0, alignment))
    
    def _calculate_resonance_potential(self, row: pd.Series) -> float:
        """Calculate resonance potential based on compound properties."""
        base_potential = 0.5
        
        # Different medical systems have different baseline resonance potentials
        if 'system' in row:
            system_bonus = {
                'Ayurvedic': 0.1,
                'Homeopathic': 0.2,
                'Unani': 0.1,
                'Allopathic': 0.05
            }
            base_potential += system_bonus.get(row['system'], 0)
            
        # Compounds with multiple effects have higher resonance
        if 'active_compounds_count' in row:
            base_potential += min(0.2, row['active_compounds_count'] * 0.02)
            
        # Add some randomness
        potential = base_potential + random.uniform(-0.1, 0.1)
        
        # Ensure within range [0, 1]
        return max(0.0, min(1.0, potential))

if __name__ == "__main__":
    # Process data when run directly
    processor = MedicinalDataProcessor()
    processed_df = processor.prepare_diabetes_dataset()
    print(f"Processed {len(processed_df)} compounds.") 