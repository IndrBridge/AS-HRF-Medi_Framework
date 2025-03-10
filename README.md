# Medicinal Harmonic Resonance AI

An AI-driven drug discovery platform that integrates traditional medical systems (Ayurveda, Unani, Allopathy, and Homeopathy) with modern computational methods to identify effective compound combinations.

## Project Overview

This platform uses artificial intelligence to:
1. Analyze compounds from diverse medical traditions
2. Predict efficacy, safety, and interactions
3. Generate optimal combinations for disease treatment
4. Validate predictions through simulation

## Current Focus

The current proof-of-concept focuses on Type 2 Diabetes treatments, integrating knowledge from four medical systems to discover synergistic compound combinations.

## Live Demo

The app is deployed on Streamlit Cloud and can be accessed here: [Medicinal Harmonic Resonance AI](https://medicinal-harmonic-ai.streamlit.app)

## Features

- **Cross-System Analysis**: Combines knowledge from Ayurveda, Unani, Allopathy, and Homeopathy
- **AI-Powered Predictions**: Uses machine learning to predict compound efficacy
- **Harmonic Resonance**: Identifies synergistic combinations across medical traditions
- **Intuitive UI**: User-friendly interface with explanations and tooltips

### New Features

- **Compound Combination Analyzer**: Test multiple compounds together to discover synergistic effects
- **Interactive Visualizations**: Dynamic charts for visualizing synergy, efficacy progression, and interactions
- **Validation Tools**: Literature-based validation with confidence scoring and experiment suggestions
- **Enhanced Prediction Interface**: More user-friendly sliders and comprehensive visual feedback
- **Safety Profiles**: Detailed safety assessments for compound combinations

## Local Development

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AS-HRF-Medi_Framework.git
   cd AS-HRF-Medi_Framework
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.9 -m venv venv39
   source venv39/bin/activate  # On Windows: venv39\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Directory Structure

- `/data`: Databases and datasets
- `/models`: ML/AI models
- `/med_harmonic_ai/scripts`: Utility scripts for data processing and modeling
- `/med_harmonic_ai/api`: Streamlit web application
- `/med_harmonic_ai/data`: Application-specific data
- `/med_harmonic_ai/models`: Application-specific models

## Deployment to Streamlit Cloud

This application is ready for deployment on Streamlit Cloud:

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account
3. Select "New app"
4. Connect your GitHub repository
5. Set the main file path to `streamlit_app.py`
6. Choose Python 3.9
7. Deploy!

## Future Enhancements

- Integration with real compound databases
- Molecular dynamics simulation
- Advanced visualization of compound interactions
- API for programmatic access 