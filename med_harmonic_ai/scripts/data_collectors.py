"""
Data collectors for the Medicinal Harmonic Resonance Framework.
This module collects compound information from various traditional medicine systems.
"""
import os
import sys
import random
import json
from typing import Dict, List, Optional, Tuple

# Set proper path for Python 3.9 environment
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

# Only after setting path, import pandas
import pandas as pd
import numpy as np

# Create data directories
def ensure_data_dirs():
    """Ensure all necessary data directories exist."""
    os.makedirs(os.path.join(BASE_DIR, 'med_harmonic_ai/data'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'med_harmonic_ai/data/raw'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'data/raw'), exist_ok=True)

# Functions
def collect_ayurvedic_compounds():
    """
    Collect compounds from Ayurvedic medicine system.
    
    Returns:
        pd.DataFrame: DataFrame containing Ayurvedic compounds
    """
    print("Collecting Ayurvedic compounds...")
    
    # For demonstration, creating synthesized data
    compounds = []
    
    ayurvedic_plants = [
        "Turmeric", "Gymnema", "Fenugreek", "Bitter Melon", 
        "Indian Kino", "Holy Basil", "Neem", "Amla",
        "Cinnamon", "Aloe Vera", "Ashwagandha", "Gudmar",
        "Guggul", "Arjuna", "Licorice", "Cardamom",
        "Triphala", "Boswellia", "Cumin", "Ginger"
    ]
    
    for plant in ayurvedic_plants:
        efficacy = random.uniform(0.6, 0.95) if plant in ["Gymnema", "Bitter Melon", "Fenugreek", "Turmeric"] else random.uniform(0.2, 0.8)
        
        compound = {
            "name": plant,
            "system": "Ayurvedic",
            "effective_for_diabetes": efficacy > 0.7,
            "is_bitter": random.choice([0, 1]),
            "is_cooling": random.choice([0, 1]),
            "reduces_kapha": random.choice([0, 1]),
            "active_compounds_count": random.randint(1, 10),
            "has_blood_sugar_effect": 1 if efficacy > 0.6 else 0,
            "efficacy_amplitude": efficacy,
            "system_frequency": 0.33,  # Specific to Ayurvedic
            "phase_alignment": random.uniform(0.3, 0.9),
            "resonance_potential": random.uniform(0.4, 0.9)
        }
        compounds.append(compound)
    
    return pd.DataFrame(compounds)

def collect_allopathic_compounds():
    """
    Collect compounds from Allopathic (conventional) medicine system.
    
    Returns:
        pd.DataFrame: DataFrame containing Allopathic compounds
    """
    print("Collecting Allopathic compounds...")
    
    # For demonstration, creating synthesized data
    compounds = []
    
    allopathic_drugs = [
        "Metformin", "Sitagliptin", "Empagliflozin", "Liraglutide", 
        "Glimepiride", "Pioglitazone", "Acarbose", "Canagliflozin",
        "Glyburide", "Rosiglitazone", "Repaglinide", "Dapagliflozin",
        "Exenatide", "Dulaglutide", "Saxagliptin", "Linagliptin",
        "Semaglutide", "Glipizide", "Nateglinide", "Insulin"
    ]
    
    for drug in allopathic_drugs:
        efficacy = random.uniform(0.75, 0.98) if drug in ["Metformin", "Insulin", "Liraglutide", "Empagliflozin"] else random.uniform(0.5, 0.85)
        
        compound = {
            "name": drug,
            "system": "Allopathic",
            "effective_for_diabetes": efficacy > 0.7,
            "half_life_hours": random.uniform(1.0, 24.0),
            "bioavailability": random.uniform(0.3, 0.95),
            "active_compounds_count": random.randint(1, 3),
            "has_blood_sugar_effect": 1,
            "efficacy_amplitude": efficacy,
            "system_frequency": 0.67,  # Specific to Allopathic
            "phase_alignment": random.uniform(0.5, 0.95),
            "resonance_potential": random.uniform(0.3, 0.8)
        }
        compounds.append(compound)
    
    return pd.DataFrame(compounds)

def collect_unani_compounds():
    """
    Collect compounds from Unani medicine system.
    
    Returns:
        pd.DataFrame: DataFrame containing Unani compounds
    """
    print("Collecting Unani compounds...")
    
    # For demonstration, creating synthesized data
    compounds = []
    
    unani_ingredients = [
        "Myrtle", "Cinnamon", "Black Seed", "Fenugreek", 
        "Ginger", "Aloe", "Cardamom", "Olive",
        "Garlic", "Gooseberry", "Turmeric", "Bitter Gourd",
        "Coriander", "Pomegranate", "Chicory", "Rose",
        "Fennel", "Barley", "Saffron", "Honey"
    ]
    
    for ingredient in unani_ingredients:
        efficacy = random.uniform(0.6, 0.9) if ingredient in ["Black Seed", "Fenugreek", "Bitter Gourd", "Cinnamon"] else random.uniform(0.3, 0.75)
        
        compound = {
            "name": ingredient,
            "system": "Unani",
            "effective_for_diabetes": efficacy > 0.7,
            "is_hot": random.choice([0, 1]),
            "reduces_phlegm": random.choice([0, 1]),
            "active_compounds_count": random.randint(1, 8),
            "has_blood_sugar_effect": 1 if efficacy > 0.65 else 0,
            "efficacy_amplitude": efficacy,
            "system_frequency": 0.5,  # Specific to Unani
            "phase_alignment": random.uniform(0.4, 0.85),
            "resonance_potential": random.uniform(0.4, 0.85)
        }
        compounds.append(compound)
    
    return pd.DataFrame(compounds)

def collect_homeopathic_compounds():
    """
    Collect compounds from Homeopathic medicine system.
    
    Returns:
        pd.DataFrame: DataFrame containing Homeopathic compounds
    """
    print("Collecting Homeopathic compounds...")
    
    # For demonstration, creating synthesized data
    compounds = []
    
    homeopathic_remedies = [
        "Uranium Nitricum", "Syzygium Jambolanum", "Phosphoric Acid", "Gymnema Sylvestre", 
        "Cephalandra Indica", "Acid Phos", "Arsenicum Album", "Lycopodium",
        "Natrum Sulph", "Insulinum", "Bryonia Alba", "Chionanthus Virginica",
        "Iris Versicolor", "Phosphorus", "Rhus Aromatica", "Allium Sativum",
        "Helleborus Niger", "Abroma Augusta", "Belladonna", "Calendula"
    ]
    
    for remedy in homeopathic_remedies:
        efficacy = random.uniform(0.5, 0.9) if remedy in ["Uranium Nitricum", "Syzygium Jambolanum", "Gymnema Sylvestre", "Cephalandra Indica"] else random.uniform(0.3, 0.7)
        
        compound = {
            "name": remedy,
            "system": "Homeopathic",
            "effective_for_diabetes": efficacy > 0.7,
            "has_thirst_symptom": random.choice([0, 1]),
            "has_urination_symptom": random.choice([0, 1]),
            "active_compounds_count": random.randint(1, 5),
            "has_blood_sugar_effect": 1 if efficacy > 0.6 else 0,
            "efficacy_amplitude": efficacy,
            "system_frequency": 0.25,  # Specific to Homeopathic
            "phase_alignment": random.uniform(0.3, 0.8),
            "resonance_potential": random.uniform(0.5, 0.9)
        }
        compounds.append(compound)
    
    return pd.DataFrame(compounds)

def save_data(df, filename):
    """
    Save DataFrame to CSV in the raw data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Filename to save as
    """
    # Save to both possible paths for compatibility
    main_path = os.path.join(BASE_DIR, 'data/raw', filename)
    app_path = os.path.join(BASE_DIR, 'med_harmonic_ai/data/raw', filename)
    
    df.to_csv(main_path, index=False)
    print(f"Data saved to {main_path}")
    
    # Also save to app directory
    df.to_csv(app_path, index=False)
    print(f"Data saved to {app_path}")

def collect_all_data():
    """
    Collect data from all medical systems and combine.
    
    Returns:
        pd.DataFrame: Combined DataFrame with compounds from all systems
    """
    # Collect data from each system
    ayurvedic_df = collect_ayurvedic_compounds()
    allopathic_df = collect_allopathic_compounds()
    unani_df = collect_unani_compounds()
    homeopathic_df = collect_homeopathic_compounds()
    
    # Combine into a single DataFrame
    all_compounds_df = pd.concat([
        ayurvedic_df,
        allopathic_df,
        unani_df,
        homeopathic_df
    ], ignore_index=True)
    
    # Save individual and combined datasets
    save_data(ayurvedic_df, 'ayurvedic_compounds.csv')
    save_data(allopathic_df, 'allopathic_compounds.csv')
    save_data(unani_df, 'unani_compounds.csv')
    save_data(homeopathic_df, 'homeopathic_compounds.csv')
    save_data(all_compounds_df, 'all_compounds.csv')
    
    return all_compounds_df

if __name__ == "__main__":
    # Create base directories
    ensure_data_dirs()
    
    # Collect all data
    all_data = collect_all_data()
    print(f"Collected data for {len(all_data)} compounds across all medical systems.")