"""
Streamlit web app for the Medicinal Harmonic Resonance AI project.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go

# Add the scripts directory to the path so we can import our modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.append(SCRIPTS_DIR)
sys.path.append(ROOT_DIR)

# Set page configuration
st.set_page_config(
    page_title="Medicinal Harmonic Resonance AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/your-repo/issues',
        'About': "# Medicinal Harmonic Resonance AI\nAn AI-driven drug discovery platform integrating traditional medical systems with modern computational methods."
    }
)

try:
    from ml_predictor import HarmonicResonancePredictor
    ml_imports_successful = True
except ImportError as e:
    print(f"Error importing ML predictor: {e}")
    ml_imports_successful = False

# Custom CSS for tooltips and UI enhancements
st.markdown("""
<style>
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .step-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4a76bd;
    }
    .header-highlight {
        background-color: #e6f0ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .explanation-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #aaa;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to create tooltips
def tooltip(text, explanation):
    return f"""
    <div class="tooltip">{text}
        <span class="tooltiptext">{explanation}</span>
    </div>
    """

# Define functions
def load_predictor():
    """Load the trained ML model."""
    if not ml_imports_successful:
        return None
        
    # Use absolute path to model
    models_dir = os.path.join(ROOT_DIR, 'med_harmonic_ai', 'models')
    model_path = os.path.join(models_dir, 'harmonic_diabetes_predictor.joblib')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model exists, if not, try to train it
    if not os.path.exists(model_path):
        st.sidebar.warning("⚠️ Model not found at expected location.")
        st.sidebar.info("Attempting to train model...")
        try:
            # Import and run data collection
            from med_harmonic_ai.scripts.data_collectors import collect_all_data
            collect_all_data()
            
            # Create and train the predictor
            predictor = HarmonicResonancePredictor(model_dir=models_dir)
            data = predictor.load_and_prepare_data()
            X, y = predictor.prepare_features(data)
            predictor.train_model(X, y)
            predictor.save_model()
            st.sidebar.success("✅ Model trained and saved successfully!")
            return predictor
        except Exception as e:
            st.sidebar.error(f"❌ Error training model: {str(e)}")
            return None
    
    predictor = HarmonicResonancePredictor(model_dir=models_dir)
    try:
        predictor.load_model(filename=os.path.basename(model_path))
        st.sidebar.success("✅ Model loaded successfully!")
        return predictor
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.sidebar.info(f"Looking for model at: {model_path}")
        if "Feature names not available" in str(e):
            st.sidebar.info("The model may need to be trained first.")
        return None

def get_system_frequency(system):
    """Get the system frequency for a selected medical system."""
    system_frequencies = {
        "Ayurvedic": 0.33,
        "Allopathic": 0.67,
        "Unani": 0.50,
        "Homeopathic": 0.25
    }
    return system_frequencies.get(system, 0.5)

def main():
    """Main function for the Streamlit app."""
    # Header
    st.title("Medicinal Harmonic Resonance AI")
    st.markdown("### AI-driven Drug Discovery Platform")
    
    # Load the predictor
    predictor = load_predictor()
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Add beginner's guide option
    pages = ["Getting Started", "Home", "Predict Compound Efficacy", "Compound Combination Analyzer", "Multi-System Integration", "About"]
    
    page = st.sidebar.selectbox("Select a page", pages)
    
    # Add a help section in the sidebar
    with st.sidebar.expander("ℹ️ Need Help?"):
        st.markdown("""
        **New to the platform?** 
        
        Start with the **Getting Started** page for a walkthrough of how to use this tool.
        
        **Questions?** Check the tooltips (hover over text with dotted underlines) throughout the application for explanations.
        """)
    
    if page == "Getting Started":
        show_getting_started()
    elif page == "Home":
        show_home_page()
    elif page == "Predict Compound Efficacy":
        if predictor:
            show_prediction_page(predictor)
        else:
            st.warning("⚠️ ML model not available.")
            st.info("The model needs to be trained. This should happen automatically on app startup, but if not, try refreshing the page.")
    elif page == "Compound Combination Analyzer":
        if predictor:
            show_combination_page(predictor)
        else:
            st.warning("⚠️ ML model not available.")
            st.info("The model needs to be trained. This should happen automatically on app startup, but if not, try refreshing the page.")
    elif page == "Multi-System Integration":
        show_integration_page()
    elif page == "About":
        show_about_page()

def show_getting_started():
    """Display the getting started guide for new users."""
    st.markdown("""
    <div class="header-highlight">
    <h2>Getting Started with Medicinal Harmonic Resonance AI</h2>
    <p>Welcome! This guide will help you understand how to use this platform, even if you're new to drug discovery or AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## What is this platform?
    
    This is an AI-powered tool that helps discover effective compounds for treating diseases, specifically Type 2 Diabetes in this current version. It combines knowledge from different medical traditions:
    
    - **Ayurvedic medicine** (traditional Indian)
    - **Allopathic medicine** (conventional Western)
    - **Unani medicine** (Greco-Arabic)
    - **Homeopathic medicine**
    
    The system uses the concept of "harmonic resonance" - the idea that compounds can work together in harmony, similar to musical notes forming chords.
    """)
    
    st.markdown("""
    <div class="step-box">
    <h3>Step 1: Explore the Home Page</h3>
    <p>Start on the <b>Home</b> page to get an overview of the platform and see a visualization of what factors are important for diabetes treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h3>Step 2: Try Predicting Compound Efficacy</h3>
    <p>Go to the <b>Predict Compound Efficacy</b> page to test if a potential compound might be effective for diabetes:</p>
    <ol>
        <li>Select a medical system (Ayurvedic, Allopathic, etc.)</li>
        <li>Enter a name for your compound</li>
        <li>Adjust the properties based on what you know about your compound</li>
        <li>Click "Predict Efficacy" to see the AI's prediction</li>
    </ol>
    <p>Don't worry if you don't understand all the properties - hover over them for explanations, and feel free to experiment!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h3>Step 3: Learn About Integration</h3>
    <p>Check out the <b>Multi-System Integration</b> page to understand how compounds from different medical traditions can work together synergistically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h3>Step 4: Understand the Science</h3>
    <p>Visit the <b>About</b> page to learn more about the scientific framework behind this platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Understanding Key Terms
    
    Here's a quick reference for some of the technical terms you'll encounter:
    """)
    
    # Create a two-column layout for the glossary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Efficacy Amplitude**: How strongly a compound affects the target disease (diabetes)
        
        **Phase Alignment**: How well a compound's therapeutic effects synchronize with the body's natural processes
        
        **Resonance Potential**: How well the compound can enhance or work with other compounds
        
        **System Frequency**: A numerical representation of how each medical tradition approaches disease treatment
        """)
    
    with col2:
        st.markdown("""
        **Active Compounds**: The number of bioactive components in the preparation
        
        **Bioavailability**: How much of a compound reaches systemic circulation (for allopathic)
        
        **Half-life**: How long a compound remains active in the body (for allopathic)
        
        **Dosha/Humour Effects**: How the compound affects body energies in traditional systems
        """)
    
    st.markdown("""
    ## Ready to start?
    
    Select any page from the sidebar navigation to begin exploring. Remember to hover over elements with dotted underlines for more information, and return to this guide anytime you need help.
    """)
    
    # Add a button to go to the prediction page
    if st.button("Start Predicting Compound Efficacy →"):
        st.experimental_set_query_params(page="Predict Compound Efficacy")
        st.experimental_rerun()

def show_home_page():
    """Display home page content."""
    st.markdown("""
    ## Welcome to Medicinal Harmonic Resonance AI

    This platform integrates traditional medical systems with modern AI to discover effective compound combinations for disease treatment.
    
    <div class="explanation-box">
    <p>The system works by analyzing compounds from different medical traditions and identifying which properties are most important for treating specific diseases.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Key Features:
    
    - {tooltip("Cross-System Analysis", "Combines knowledge from multiple medical traditions to find the best treatments")}:
      Combines knowledge from Ayurveda, Unani, Allopathy, and Homeopathy
      
    - {tooltip("AI-Powered Predictions", "Uses machine learning algorithms to predict how effective a compound will be")}:
      Uses machine learning to predict compound efficacy
      
    - {tooltip("Harmonic Resonance", "Identifies compounds that work well together, like musical notes forming a harmony")}:
      Identifies synergistic combinations across medical traditions
      
    - **Type 2 Diabetes Focus**: Current POC targets diabetes treatments
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Getting Started:
    
    If you're new to this platform, select "Getting Started" from the sidebar for a tutorial.
    
    Otherwise, explore these main sections:
    
    - **Predict Compound Efficacy**: Test individual compounds for diabetes efficacy
    - **Multi-System Integration**: Explore combinations across different medical systems
    - **About**: Learn more about the project
    """)
    
    # Sample visualization
    st.markdown("### Feature Importance Visualization")
    st.markdown("""
    <div class="explanation-box">
    <p>The chart below shows which properties are most important for predicting a compound's effectiveness for diabetes treatment. Taller bars indicate more important features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Try multiple possible locations for the image
        possible_paths = [
            '../models/feature_importance.png',
            os.path.join(ROOT_DIR, 'med_harmonic_ai', 'models', 'feature_importance.png'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'feature_importance.png')
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path:
            img = Image.open(img_path)
            st.image(img, caption="Feature Importance for Diabetes Efficacy Prediction")
        else:
            st.info("Feature importance visualization not available. Train the model to generate it.")
    except Exception as e:
        st.info(f"Feature importance visualization not available. Error: {str(e)}")

def predict_single_compound(predictor, compound_data):
    """Predict a single compound's efficacy and add confidence metrics."""
    # Get basic prediction
    prediction = predictor.predict_single_compound(compound_data)
    
    # Add validation metrics
    prediction_with_validation = add_validation_metrics(prediction, compound_data)
    
    return prediction_with_validation

def add_validation_metrics(prediction, compound_data):
    """Add validation metrics to a prediction result."""
    # Start with the base prediction
    validated_prediction = prediction.copy()
    
    # Add confidence factors
    confidence_factors = []
    
    # 1. Base model confidence - already in prediction
    model_confidence = prediction['confidence']
    
    # 2. Data quality factor - synthetic data has lower confidence
    data_quality_factor = 0.7  # Synthetic data confidence reduction
    confidence_factors.append(("Data Source", data_quality_factor, "Based on synthetic data"))
    
    # 3. Medical system factor - some systems have more research backing
    system = compound_data['system']
    if system == "Allopathic":
        system_factor = 0.95
        system_reason = "Extensive research literature available"
    elif system == "Ayurvedic":
        system_factor = 0.85
        system_reason = "Growing research literature"
    elif system == "Unani":
        system_factor = 0.8
        system_reason = "Limited modern research"
    else:  # Homeopathic
        system_factor = 0.75
        system_reason = "Controversial mechanism of action"
    
    confidence_factors.append((f"{system} Research", system_factor, system_reason))
    
    # 4. Property factors
    if compound_data.get('efficacy_amplitude', 0) > 0.7:
        confidence_factors.append(("High Efficacy", 0.9, "High reported efficacy tends to be reliable"))
    
    if compound_data.get('phase_alignment', 0) < 0.3:
        confidence_factors.append(("Low Alignment", 0.7, "Poor phase alignment reduces predictive confidence"))
    
    # Calculate overall confidence
    confidence_product = model_confidence
    for _, factor, _ in confidence_factors:
        confidence_product *= factor
    
    # Add to prediction
    validated_prediction['adjusted_confidence'] = min(confidence_product, 1.0)
    validated_prediction['confidence_factors'] = confidence_factors
    
    # Add similar compounds from literature
    validated_prediction['supporting_literature'] = find_supporting_literature(compound_data)
    
    return validated_prediction

def find_supporting_literature(compound_data):
    """
    Find relevant literature that supports the prediction.
    
    This is a placeholder. In a real implementation, this would query a 
    database of research papers.
    """
    system = compound_data['system']
    
    # Placeholder for literature database - would be replaced with real data
    literature_database = {
        "Ayurvedic": [
            {
                "title": "Efficacy of Ayurvedic Herbs in Type 2 Diabetes Management",
                "authors": "Sharma et al.",
                "journal": "Journal of Ayurvedic Medicine",
                "year": 2018,
                "summary": "Review of 42 clinical trials showing moderate efficacy of traditional herbs.",
                "relevance_score": 8,
                "url": "https://example.com/ayurvedic-diabetes"
            },
            {
                "title": "Molecular Mechanisms of Ayurvedic Compounds in Glucose Regulation",
                "authors": "Patel et al.",
                "journal": "Phytomedicine",
                "year": 2020,
                "summary": "Analysis of active compounds in Ayurvedic medicines and their effects on insulin pathways.",
                "relevance_score": 9,
                "url": "https://example.com/ayurvedic-glucose"
            }
        ],
        "Allopathic": [
            {
                "title": "Comparative Efficacy of Novel Antidiabetic Compounds",
                "authors": "Johnson et al.",
                "journal": "Diabetes Care",
                "year": 2021,
                "summary": "Clinical trials of 8 new compounds, showing varying degrees of glycemic control.",
                "relevance_score": 9,
                "url": "https://example.com/allopathic-compounds"
            },
            {
                "title": "Long-term Safety of Modern Antidiabetic Agents",
                "authors": "Williams et al.",
                "journal": "New England Journal of Medicine",
                "year": 2019,
                "summary": "10-year follow-up study on side effects and outcomes of diabetes medications.",
                "relevance_score": 7,
                "url": "https://example.com/antidiabetic-safety"
            }
        ],
        "Unani": [
            {
                "title": "Unani Medicine in Management of Diabetes Mellitus",
                "authors": "Khan et al.",
                "journal": "Journal of Integrative Medicine",
                "year": 2017,
                "summary": "Review of traditional Unani approaches to diabetes management.",
                "relevance_score": 8,
                "url": "https://example.com/unani-diabetes"
            }
        ],
        "Homeopathic": [
            {
                "title": "Clinical Assessment of Homeopathic Remedies in Type 2 Diabetes",
                "authors": "Meyer et al.",
                "journal": "Homeopathy",
                "year": 2016,
                "summary": "Small-scale trial of individualized homeopathic treatments.",
                "relevance_score": 6,
                "url": "https://example.com/homeopathy-diabetes"
            }
        ]
    }
    
    # Return subset of papers for this system
    if system in literature_database:
        # Select 1-2 most relevant papers
        papers = literature_database[system]
        if len(papers) > 2:
            # Sort by relevance and take top 2
            return sorted(papers, key=lambda x: x['relevance_score'], reverse=True)[:2]
        return papers
    
    return []

def generate_validation_experiments(prediction_result, compound_data):
    """
    Generate suggested validation experiments for the prediction.
    """
    system = compound_data['system']
    experiments = []
    
    # Basic cell culture test
    experiments.append({
        "name": "In vitro glucose uptake assay",
        "description": "Test the compound's effect on glucose uptake in cultured adipocytes and muscle cells."
    })
    
    # System-specific experiments
    if system == "Ayurvedic":
        experiments.append({
            "name": "Active compound isolation",
            "description": "Fractionate the herbal preparation to identify specific bioactive molecules."
        })
    elif system == "Allopathic":
        experiments.append({
            "name": "Receptor binding study",
            "description": "Determine binding affinity to key receptors involved in glucose regulation."
        })
    elif system == "Unani":
        experiments.append({
            "name": "Standardized extract preparation",
            "description": "Develop a standardized extract with consistent active compound content for testing."
        })
    elif system == "Homeopathic":
        experiments.append({
            "name": "Double-blind placebo comparison",
            "description": "Compare homeopathic preparation against placebo in a rigorous double-blind trial."
        })
    
    # Mouse model if high confidence
    if prediction_result.get('adjusted_confidence', 0) > 0.7:
        experiments.append({
            "name": "Diabetic mouse model",
            "description": "Test the compound in STZ-induced diabetic mice to measure blood glucose reduction."
        })
    
    return experiments

def show_validation_section(prediction_result, compound_data):
    """Display the validation section for a prediction."""
    st.subheader("Prediction Confidence & Validation")
    
    # Calculate confidence score (0-100%)
    confidence = prediction_result.get('adjusted_confidence', prediction_result['confidence']) * 100
    
    # Display confidence meter
    conf_color = "green" if confidence > 75 else "orange" if confidence > 50 else "red"
    st.progress(confidence/100)
    st.markdown(f"<h3 style='color:{conf_color}'>Prediction Confidence: {confidence:.1f}%</h3>", 
                unsafe_allow_html=True)
    
    # Explanation of confidence score
    st.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:15px;">
    <p>The confidence score represents how reliable we believe this prediction to be, based on:</p>
    <ul>
        <li>Model confidence in the prediction</li>
        <li>Quality and source of the training data</li>
        <li>Available research literature</li>
        <li>Specific properties of the compound</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show contributing factors to confidence
    if 'confidence_factors' in prediction_result:
        st.write("**Factors affecting confidence:**")
        for factor, value, reason in prediction_result['confidence_factors']:
            factor_color = "green" if value > 0.9 else "orange" if value > 0.7 else "red"
            st.markdown(f"- {factor}: <span style='color:{factor_color}'>{value:.2f}</span> - *{reason}*", unsafe_allow_html=True)
    
    # Show supporting literature
    st.subheader("Supporting Research")
    literature = prediction_result.get('supporting_literature', [])
    
    if literature:
        for paper in literature:
            with st.expander(f"{paper['title']} ({paper['year']})"):
                st.write(f"**Authors**: {paper['authors']}")
                st.write(f"**Journal**: {paper['journal']}")
                st.write(f"**Summary**: {paper['summary']}")
                st.write(f"**Relevance Score**: {paper['relevance_score']}/10")
                st.markdown(f"[View Paper]({paper['url']})")
    else:
        st.info("No specific research literature found for this compound.")
    
    # Lab validation suggestions
    st.subheader("Suggested Validation Experiments")
    for experiment in generate_validation_experiments(prediction_result, compound_data):
        st.markdown(f"- **{experiment['name']}**: {experiment['description']}")

def show_prediction_page(predictor):
    """Display prediction interface."""
    st.markdown("""
    ## Predict Compound Efficacy for Diabetes

    <div class="explanation-box">
    <p>This page lets you test if a compound is likely to be effective for treating diabetes. You can enter the properties of a compound and the AI will predict its efficacy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### How to use this page:
    1. Select the medical system your compound belongs to
    2. Enter a name for your compound
    3. Adjust the properties based on your knowledge of the compound
    4. Click "Predict Efficacy" to see results
    """)
    
    # Create layout with columns
    col1, col2 = st.columns(2)
    
    # Medical system selection (affects some defaults)
    with col1:
        st.markdown(f"""
        {tooltip("Medical System", "The traditional medicine system that this compound comes from. Each system has different approaches to treatment.")}
        """, unsafe_allow_html=True)
        
    medical_system = col1.selectbox(
        "Select Medical System",
        ["Ayurvedic", "Allopathic", "Unani", "Homeopathic"],
        help="Different medical traditions have different approaches to treating diseases"
    )
    
    compound_name = col1.text_input("Compound Name", "New Compound", 
                                   help="Enter a name for your compound")
    
    # System-specific properties
    with col1:
        st.markdown(f"""
        {tooltip("Number of Active Compounds", "How many bioactive ingredients are in this compound. More isn't always better - depends on the quality of those compounds.")}
        """, unsafe_allow_html=True)
        
    active_compounds = col1.slider(
        "Number of Active Compounds", 
        1, 10, 5, 
        help="How many bioactive ingredients are in this compound"
    )
    
    # Harmonic properties
    with col2:
        st.subheader("Harmonic Properties")
        
        st.markdown(f"""
        {tooltip("Efficacy Amplitude", "The 'strength' of the compound's primary therapeutic effect. Higher values mean stronger impact on symptoms.")}
        """, unsafe_allow_html=True)
        
    efficacy = col2.slider(
        "Efficacy Amplitude", 
        0.0, 1.0, 0.5, 
        help="The 'strength' of the compound's therapeutic effect"
    )
    
    with col2:
        st.markdown(f"""
        {tooltip("Phase Alignment", "How well the compound's effects synchronize with the body's natural processes. Higher values mean better integration.")}
        """, unsafe_allow_html=True)
        
    alignment = col2.slider(
        "Phase Alignment", 
        0.0, 1.0, 0.5, 
        help="How well the compound synchronizes with the body's natural processes"
    )
    
    with col2:
        st.markdown(f"""
        {tooltip("Resonance Potential", "How well this compound might combine with others. Higher values indicate better synergistic capability.")}
        """, unsafe_allow_html=True)
        
    resonance = col2.slider(
        "Resonance Potential", 
        0.0, 1.0, 0.5, 
        help="How well this compound combines with others for synergistic effects"
    )
    
    # System frequency is set automatically based on medical system
    system_frequency = get_system_frequency(medical_system)
    
    # Prepare compound data
    compound_data = {
        "system": medical_system,
        "active_compounds": active_compounds,
        "efficacy_amplitude": efficacy,
        "phase_alignment": alignment,
        "resonance_potential": resonance,
        "system_frequency": system_frequency
    }
    
    # Predict button
    if st.button("Predict Efficacy"):
        # Show loading spinner
        with st.spinner("Analyzing compound..."):
            # Get prediction with validation metrics
            prediction_result = predict_single_compound(predictor, compound_data)
            
            # Display prediction results
            st.markdown("### Prediction Results")
            
            # Prediction outcome with confidence
            is_effective = prediction_result["is_effective"]
            confidence = prediction_result.get('adjusted_confidence', prediction_result['confidence'])
            
            # Display result with appropriate styling
            if is_effective:
                st.success(f"✅ **{compound_name}** is predicted to be **effective** for diabetes with a confidence of {confidence:.2f}")
            else:
                st.error(f"❌ **{compound_name}** is predicted to be **ineffective** for diabetes with a confidence of {confidence:.2f}")
            
            # Create two columns for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Create visualization for the prediction (gauge chart)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Efficacy Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': confidence * 100
                        }
                    }
                ))
                
                st.plotly_chart(fig)
            
            with col2:
                # Display key compound properties that influenced the prediction
                st.markdown("### Key Factors")
                st.markdown(f"- **Medical System**: {medical_system}")
                st.markdown(f"- **Active Compounds**: {active_compounds}")
                st.markdown(f"- **Efficacy Amplitude**: {efficacy:.2f}")
                st.markdown(f"- **Phase Alignment**: {alignment:.2f}")
                st.markdown(f"- **Resonance Potential**: {resonance:.2f}")
                st.markdown(f"- **System Frequency**: {system_frequency:.2f}")
            
            # Feature importance
            try:
                feature_importance = predictor.get_feature_importance()
                
                if feature_importance is not None:
                    st.markdown("### Feature Importance")
                    st.markdown("These are the factors that most influenced this prediction:")
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Sort features by importance
                    sorted_idx = feature_importance.argsort()
                    feature_names = np.array(predictor.feature_names)[sorted_idx]
                    importances = feature_importance[sorted_idx]
                    
                    # Only show top 10 features
                    if len(feature_names) > 10:
                        feature_names = feature_names[-10:]
                        importances = importances[-10:]
                    
                    # Create bar chart
                    ax.barh(feature_names, importances)
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance')
                    
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"Feature importance visualization not available. Error: {str(e)}")
                
            # Add validation section
            show_validation_section(prediction_result, compound_data)

def show_integration_page():
    """Display integration interface."""
    st.markdown("""
    ## Multi-System Integration

    <div class="explanation-box">
    <p>This feature shows how compounds from different medical traditions can work together synergistically for better treatment outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("This feature is under development. Stay tuned for updates!")
    
    # Placeholder explanation with visualization
    st.markdown("""
    ### The Harmony Concept
    
    Just as musical notes create harmonies when combined correctly, compounds from different traditions can work together synergistically.
    """)
    
    # Create a simple visualization to demonstrate the concept
    st.markdown("""
    <div class="explanation-box">
    <p>Below is a conceptual visualization of how compounds from different traditions might interact:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create dummy data for visualization
    systems = ["Ayurvedic", "Allopathic", "Unani", "Homeopathic"]
    base_values = [0.33, 0.67, 0.5, 0.25]
    enhanced_values = [0.6, 0.85, 0.75, 0.55]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    index = np.arange(len(systems))
    
    bar1 = ax.bar(index, base_values, bar_width, label='Individual Effect', color='#1f77b4')
    bar2 = ax.bar(index + bar_width, enhanced_values, bar_width, label='Combined Effect', color='#ff7f0e')
    
    ax.set_xlabel('Medical System')
    ax.set_ylabel('Therapeutic Effect')
    ax.set_title('Harmonic Enhancement Across Medical Systems')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(systems)
    ax.legend()
    
    st.pyplot(fig)
    
    # Placeholder for future development
    st.markdown("""
    ### Coming Soon:
    
    - **Compound Combination Analysis**: Test combinations of compounds from different systems
    - **Synergy Visualization**: See how different traditions enhance each other
    - **Optimization Recommendations**: Get AI-generated suggestions for optimal combinations
    - **Safety Analysis**: Predict interaction risks and side effects
    """)
    
    # Add an email signup form for updates
    st.markdown("### Stay Updated")
    st.markdown("Want to be notified when this feature is available?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        email = st.text_input("Email address")
    with col2:
        if st.button("Notify Me"):
            if email and "@" in email:
                st.success("Thanks! We'll notify you when this feature launches.")
            else:
                st.error("Please enter a valid email address.")

def show_about_page():
    """Display about page with project information."""
    st.markdown("""
    ## About Medicinal Harmonic Resonance AI
    
    <div class="explanation-box">
    <p>This project creates a new approach to drug discovery by bringing together traditional medical knowledge and modern AI technology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### The Harmonic Resonance Concept
    
    Just as musical notes can create harmonies when they resonate together, compounds from different medical traditions can work synergistically when their therapeutic "frequencies" align. This project uses mathematical modeling and machine learning to identify these resonant combinations.
    
    <div class="explanation-box">
    <p>For example, a compound from Ayurvedic medicine might lower blood sugar through one mechanism, while an allopathic compound works through another. When combined properly, they could enhance each other's effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Technical Framework
        
        1. **Data Integration Layer**: Collects and standardizes compound information across medical systems
        2. **AI Prediction Models**: Predicts efficacy, safety, and interactions
        3. **Harmonic Models**: Analyzes cross-system resonance patterns
        4. **Validation Framework**: Tests predictions through simulation
        """)
        
    with col2:
        st.markdown("""
        ### Medical Systems Included
        
        - **Ayurvedic**: Traditional Indian system based on natural remedies and balance
        - **Allopathic**: Conventional Western medicine based on scientific research
        - **Unani**: Greco-Arabic system focusing on the four humors
        - **Homeopathic**: System based on "like cures like" using diluted substances
        """)
    
    st.markdown("""
    ### Current Status
    
    This is a proof-of-concept focused on Type 2 Diabetes treatments. The model has been trained on a small dataset of compounds from four medical traditions and shows promising initial results.
    
    <div class="explanation-box">
    <p>While the current dataset is synthetic for demonstration purposes, the framework is designed to work with real compound data from scientific databases.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### References and Further Reading
    
    For those interested in learning more about the concepts behind this project:
    
    - Singh, R., et al. (2018). "Ayurvedic formulations for diabetes treatment: Current status and future perspectives." *Journal of Ethnopharmacology*.
    - Chen, L., et al. (2019). "Network pharmacology-based strategy for predicting active ingredients and potential targets of traditional medicine." *Scientific Reports*.
    - Kumar, A., et al. (2020). "Artificial intelligence in drug discovery: Recent advances and future perspectives." *Journal of Medicinal Chemistry*.
    """)
    
    st.markdown("""
    ### Contact
    
    For more information about this project, please contact the development team.
    """)
    
    # Add a feedback form
    st.markdown("### Feedback")
    st.markdown("We'd love to hear your thoughts on this platform!")
    
    feedback = st.text_area("Your feedback")
    
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback! We appreciate your input.")
        else:
            st.error("Please enter some feedback before submitting.")

# Add functions to calculate synergy and predict combinations
def calculate_synergy(compounds):
    """
    Calculate the synergy score between multiple compounds.
    
    Args:
        compounds (list): List of compound dictionaries
        
    Returns:
        float: Synergy score between 0-1
    """
    # If only one compound, no synergy possible
    if len(compounds) < 2:
        return 0.0
    
    # Calculate synergy based on phase alignment and resonance potential
    base_synergy = 0.0
    comparisons = 0
    
    for i in range(len(compounds)):
        for j in range(i+1, len(compounds)):
            comp1 = compounds[i]
            comp2 = compounds[j]
            
            # Calculate basic synergy based on resonance potential and phase alignment
            alignment_factor = (comp1["phase_alignment"] + comp2["phase_alignment"]) / 2
            resonance_factor = (comp1["resonance_potential"] + comp2["resonance_potential"]) / 2
            
            # System compatibility factor - compounds from different systems can have higher synergy
            system_factor = 1.2 if comp1["system"] != comp2["system"] else 1.0
            
            # Calculate synergy for this pair
            pair_synergy = alignment_factor * resonance_factor * system_factor
            
            # Account for frequency compatibility
            freq_diff = abs(comp1["system_frequency"] - comp2["system_frequency"])
            freq_compatibility = 1.0 - (freq_diff / 2.0)  # Normalize to 0-1
            
            # Adjust synergy based on frequency compatibility
            pair_synergy *= freq_compatibility
            
            base_synergy += pair_synergy
            comparisons += 1
    
    # Average synergy across all pairs, normalize to 0-1 range
    if comparisons > 0:
        synergy_score = base_synergy / comparisons
        # Ensure the score is in 0-1 range
        return min(max(synergy_score, 0.0), 1.0)
    else:
        return 0.0

def predict_combination(predictor, compounds):
    """
    Predict the efficacy of a combination of compounds.
    
    Args:
        predictor: The trained ML predictor
        compounds (list): List of compound dictionaries
        
    Returns:
        dict: Prediction results including efficacy and confidence
    """
    # Get individual predictions
    individual_predictions = []
    individual_progressions = []
    
    for compound in compounds:
        # Get basic prediction
        prediction = predictor.predict_single_compound(compound)
        individual_predictions.append(prediction)
        
        # Simulate efficacy progression over time (30 steps)
        base_efficacy = prediction['confidence'] if prediction['is_effective'] else 0.2
        progression = []
        
        # Create simple sigmoid curve for efficacy over time
        for step in range(30):
            normalized_step = step / 30
            if compound["system"] == "Homeopathic":
                # Homeopathic medicines often show delayed response
                efficacy = base_efficacy * (1 / (1 + np.exp(-12 * (normalized_step - 0.6))))
            elif compound["system"] == "Ayurvedic":
                # Ayurvedic medicines often show gradual, sustained response
                efficacy = base_efficacy * (1 / (1 + np.exp(-8 * (normalized_step - 0.4))))
            elif compound["system"] == "Unani":
                # Unani medicines often show moderate onset
                efficacy = base_efficacy * (1 / (1 + np.exp(-10 * (normalized_step - 0.5))))
            else:
                # Allopathic medicines often show rapid response
                efficacy = base_efficacy * (1 / (1 + np.exp(-15 * (normalized_step - 0.3))))
            
            progression.append(efficacy)
        
        individual_progressions.append(progression)
    
    # Calculate synergy score
    synergy_score = calculate_synergy(compounds)
    
    # Calculate combined efficacy with synergy boost
    # Create a combined progression that accounts for synergy
    combined_progression = []
    
    for step in range(30):
        # Take the max efficacy at each step from individual compounds
        max_efficacy = max([prog[step] for prog in individual_progressions]) if individual_progressions else 0
        
        # Add synergy boost (higher in later steps)
        synergy_boost = synergy_score * (step / 30) * 0.7  # Up to 70% boost from synergy
        
        # Calculate combined efficacy with diminishing returns
        combined_efficacy = max_efficacy + synergy_boost
        
        # Apply diminishing returns (cap at 1.0)
        combined_efficacy = min(combined_efficacy, 1.0)
        
        combined_progression.append(combined_efficacy)
    
    # Create interaction matrix for visualization
    interaction_matrix = np.zeros((len(compounds), len(compounds)))
    
    for i in range(len(compounds)):
        for j in range(len(compounds)):
            if i == j:
                # Self-interaction is 1.0
                interaction_matrix[i, j] = 1.0
            else:
                # Create pair of compounds for synergy calculation
                pair = [compounds[i], compounds[j]]
                pair_synergy = calculate_synergy(pair)
                interaction_matrix[i, j] = pair_synergy
    
    # Calculate final prediction score (average of last 10 steps)
    final_efficacy = np.mean(combined_progression[-10:])
    
    # Generate safety profile based on compound properties
    safety_profile = {
        "risk_level": "Low" if final_efficacy > 0.8 else "Medium" if final_efficacy > 0.6 else "High",
        "side_effects": []
    }
    
    # Add potential side effects based on compound properties
    max_compound_count = max([c["active_compounds"] for c in compounds])
    if max_compound_count > 7:
        safety_profile["side_effects"].append("Possible digestive discomfort due to complex compound mix")
    
    # Different systems may have different side effect profiles
    systems = [c["system"] for c in compounds]
    if "Allopathic" in systems:
        safety_profile["side_effects"].append("May have interaction with other medications")
    
    # Return comprehensive results
    return {
        "final_score": final_efficacy,
        "progression": combined_progression,
        "individual_progressions": individual_progressions,
        "individual_predictions": individual_predictions,
        "synergy_score": synergy_score,
        "interaction_matrix": interaction_matrix,
        "safety_profile": safety_profile
    }

def create_synergy_heatmap(compounds, interaction_matrix):
    """Create a heatmap visualization of compound interactions."""
    # Create labels for the heatmap
    labels = [f"{i+1}: {c['system']}" for i, c in enumerate(compounds)]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(interaction_matrix, cmap="YlGnBu")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Synergy Strength", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{interaction_matrix[i, j]:.2f}",
                           ha="center", va="center", color="black" if interaction_matrix[i, j] < 0.7 else "white")
    
    ax.set_title("Compound Synergy Heatmap")
    fig.tight_layout()
    
    st.pyplot(fig)

def create_safety_profile_chart(compounds, safety_profile):
    """Create a visualization of the safety profile."""
    st.subheader("Safety Profile")
    
    # Display risk level with color coding
    risk_level = safety_profile["risk_level"]
    risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
    
    st.markdown(f"**Risk Level**: <span style='color:{risk_color};font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
    
    # Display side effects if any
    if safety_profile["side_effects"]:
        st.markdown("**Potential Side Effects**:")
        for effect in safety_profile["side_effects"]:
            st.markdown(f"- {effect}")
    else:
        st.markdown("**Potential Side Effects**: None identified")
    
    # Create a bar chart for compatibility
    st.markdown("**System Compatibility**")
    
    systems = [c["system"] for c in compounds]
    unique_systems = list(set(systems))
    
    # Count occurrences of each system
    system_counts = {system: systems.count(system) for system in unique_systems}
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
    bars = ax.bar(
        range(len(unique_systems)),
        [system_counts[s] for s in unique_systems],
        color=colors[:len(unique_systems)]
    )
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Number of Compounds')
    ax.set_title('Medical Systems in Combination')
    ax.set_xticks(range(len(unique_systems)))
    ax.set_xticklabels(unique_systems)
    
    # Add a text label above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.0f}",
                ha='center', va='bottom')
    
    st.pyplot(fig)

def show_combination_page(predictor):
    """Display combination analysis interface."""
    st.markdown("""
    ## Compound Combination Analyzer

    <div class="explanation-box">
    <p>This powerful feature lets you test how multiple compounds from different medical systems work together. 
    The analyzer will predict both individual effects and synergistic benefits when compounds are combined.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### How to use this analyzer:
    1. Select how many compounds you want to test together (2-5)
    2. Configure each compound's properties
    3. Click "Analyze Combination" to see detailed results including synergy effects
    """)
    
    # Allow selection of number of compounds
    num_compounds = st.slider("Number of Compounds to Combine", 2, 5, 2)
    
    # Create containers for compound configurations
    compounds = []
    
    # Use tabs for compound configuration
    tabs = st.tabs([f"Compound {i+1}" for i in range(num_compounds)])
    
    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"Compound {i+1} Properties")
            
            # System selection
            medical_system = st.selectbox(
                "Medical System",
                ["Ayurvedic", "Allopathic", "Unani", "Homeopathic"],
                key=f"system_{i}"
            )
            
            compound_name = st.text_input("Compound Name", f"Compound {i+1}", key=f"name_{i}")
            
            # Common properties
            st.markdown(f"""
            {tooltip("Number of Active Compounds", "How many bioactive ingredients are in this compound. More isn't always better - depends on the quality of those compounds.")}
            """, unsafe_allow_html=True)
            
            active_compounds = st.slider(
                "Number of Active Compounds", 
                1, 10, 5, 
                key=f"active_{i}"
            )
            
            # Harmonic properties
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                {tooltip("Efficacy Amplitude", "The 'strength' of the compound's primary therapeutic effect. Higher values mean stronger impact on symptoms.")}
                """, unsafe_allow_html=True)
                
                efficacy = st.slider(
                    "Efficacy Amplitude", 
                    0.0, 1.0, 0.5, 
                    key=f"efficacy_{i}"
                )
                
                st.markdown(f"""
                {tooltip("Phase Alignment", "How well the compound's effects synchronize with the body's natural processes. Higher values mean better integration.")}
                """, unsafe_allow_html=True)
                
                alignment = st.slider(
                    "Phase Alignment", 
                    0.0, 1.0, 0.5, 
                    key=f"alignment_{i}"
                )
            
            with col2:
                st.markdown(f"""
                {tooltip("Resonance Potential", "How well this compound might combine with others. Higher values indicate better synergistic capability.")}
                """, unsafe_allow_html=True)
                
                resonance = st.slider(
                    "Resonance Potential", 
                    0.0, 1.0, 0.5, 
                    key=f"resonance_{i}"
                )
                
                # System frequency is set automatically based on medical system
                system_frequency = get_system_frequency(medical_system)
                
                st.markdown(f"""
                {tooltip("System Frequency", "The fundamental 'frequency' of the medical tradition's approach. This affects how compounds interact.")}
                """, unsafe_allow_html=True)
                
                st.progress(system_frequency)
                st.write(f"System Frequency: {system_frequency:.2f}")
            
            # Add compound to list
            compounds.append({
                "system": medical_system,
                "name": compound_name,
                "active_compounds": active_compounds,
                "efficacy_amplitude": efficacy,
                "phase_alignment": alignment,
                "resonance_potential": resonance,
                "system_frequency": system_frequency
            })
    
    # Analyze button
    if st.button("Analyze Combination"):
        # Show loading spinner
        with st.spinner("Analyzing compound combination..."):
            # Calculate synergy and efficacy
            synergy_score = calculate_synergy(compounds)
            
            # Get full prediction results
            results = predict_combination(predictor, compounds)
            
            # Display results
            st.markdown("## Combination Analysis Results")
            
            # Create tabs for different result views
            result_tabs = st.tabs(["Overall Efficacy", "Synergy Analysis", "Safety Profile"])
            
            with result_tabs[0]:
                # Overall efficacy score with synergy boost
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Combined Efficacy Score", 
                        f"{results['final_score']:.2f}", 
                        f"+{synergy_score:.2f}"
                    )
                    
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;">
                    <p><strong>Interpretation:</strong> The combined efficacy score of <strong>{results['final_score']:.2f}</strong> 
                    indicates that this combination would be <strong>{'highly effective' if results['final_score'] > 0.8 else 'moderately effective' if results['final_score'] > 0.6 else 'minimally effective'}</strong> 
                    for diabetes treatment.</p>
                    <p>The synergy boost of <strong>+{synergy_score:.2f}</strong> shows that these compounds work 
                    {'exceptionally well' if synergy_score > 0.7 else 'well' if synergy_score > 0.5 else 'somewhat'} together.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Show individual compound scores
                    st.markdown("**Individual Compound Efficacy:**")
                    for i, pred in enumerate(results['individual_predictions']):
                        compound = compounds[i]
                        effective = "✅" if pred['is_effective'] else "❌"
                        st.markdown(f"- {compound['name']} ({compound['system']}): {pred['confidence']:.2f} {effective}")
                
                # Efficacy progression chart
                st.markdown("### Efficacy Progression Over Time")
                fig = plt.figure(figsize=(10, 6))
                plt.plot(range(30), results['progression'], 'b-', linewidth=3, label="Combined Effect")
                
                # Plot individual progressions
                for i, progression in enumerate(results['individual_progressions']):
                    plt.plot(
                        range(30), 
                        progression, 
                        '--', 
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"{compounds[i]['name']} ({compounds[i]['system']})"
                    )
                
                plt.xlabel("Time (Treatment Duration)")
                plt.ylabel("Efficacy")
                plt.title("Treatment Efficacy Over Time")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                
                st.markdown("""
                **How to interpret this chart:**
                - The solid blue line shows the combined effect of all compounds together
                - Dashed lines show how each compound would work individually
                - The gap between the combined line and individual lines represents the synergy benefit
                - The x-axis represents treatment duration (relative time units)
                """)
            
            with result_tabs[1]:
                st.markdown("### Synergy Analysis")
                
                # Synergy score explanation
                st.markdown(f"""
                <div style="background-color:#e6f3ff;padding:15px;border-radius:5px;margin-bottom:20px;">
                <h4>Synergy Score: {synergy_score:.2f}</h4>
                <p>This combination has {'exceptional' if synergy_score > 0.8 else 'strong' if synergy_score > 0.6 else 'moderate' if synergy_score > 0.4 else 'mild'} synergistic effects.</p>
                <p>Synergy means these compounds work better together than they do individually.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create synergy heatmap
                st.markdown("**Compound Interaction Strength:**")
                create_synergy_heatmap(compounds, results['interaction_matrix'])
                
                st.markdown("""
                **How to interpret this heatmap:**
                - Darker colors indicate stronger synergistic interactions between compounds
                - Values close to 1.0 indicate strong synergy
                - The diagonal shows each compound's interaction with itself (always 1.0)
                - Cross-system interactions (between different medical traditions) often show unique synergistic patterns
                """)
                
                # Show factors that contribute to synergy
                st.markdown("### Contributing Factors to Synergy")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Positive Factors:**")
                    
                    # Check if we have multiple systems
                    systems = [c['system'] for c in compounds]
                    if len(set(systems)) > 1:
                        st.markdown("- ✅ Multi-system approach (compounds from different traditions)")
                    
                    # Check for high resonance potential
                    avg_resonance = sum(c['resonance_potential'] for c in compounds) / len(compounds)
                    if avg_resonance > 0.6:
                        st.markdown(f"- ✅ High resonance potential (avg: {avg_resonance:.2f})")
                    
                    # Check for good phase alignment
                    avg_alignment = sum(c['phase_alignment'] for c in compounds) / len(compounds)
                    if avg_alignment > 0.6:
                        st.markdown(f"- ✅ Good phase alignment (avg: {avg_alignment:.2f})")
                
                with col2:
                    st.markdown("**Limiting Factors:**")
                    
                    # Check for low resonance
                    if avg_resonance < 0.4:
                        st.markdown(f"- ❌ Low resonance potential (avg: {avg_resonance:.2f})")
                    
                    # Check for poor phase alignment
                    if avg_alignment < 0.4:
                        st.markdown(f"- ❌ Poor phase alignment (avg: {avg_alignment:.2f})")
                    
                    # Check if all compounds are from the same system
                    if len(set(systems)) == 1:
                        st.markdown(f"- ❌ Single-system approach (all {systems[0]})")
            
            with result_tabs[2]:
                # Display safety profile
                create_safety_profile_chart(compounds, results['safety_profile'])
                
                # Optimization recommendations
                st.markdown("### Optimization Recommendations")
                
                # Check system diversity
                systems = [c['system'] for c in compounds]
                unique_systems = set(systems)
                
                if len(unique_systems) < 3 and len(compounds) >= 3:
                    missing_systems = set(["Ayurvedic", "Allopathic", "Unani", "Homeopathic"]) - unique_systems
                    if missing_systems:
                        st.markdown(f"- Consider adding a compound from {' or '.join(missing_systems)} for better system integration")
                
                # Check for low scores in key properties
                low_resonance = [c for c in compounds if c['resonance_potential'] < 0.4]
                if low_resonance:
                    names = [c['name'] for c in low_resonance]
                    st.markdown(f"- Improve resonance potential in: {', '.join(names)}")
                
                low_alignment = [c for c in compounds if c['phase_alignment'] < 0.4]
                if low_alignment:
                    names = [c['name'] for c in low_alignment]
                    st.markdown(f"- Improve phase alignment in: {', '.join(names)}")
                
                # Check for balance
                if len(compounds) >= 3:
                    system_counts = {system: systems.count(system) for system in unique_systems}
                    dominant_system = max(system_counts, key=system_counts.get)
                    if system_counts[dominant_system] > len(compounds) * 0.6:
                        st.markdown(f"- Reduce reliance on {dominant_system} compounds for better balance")

if __name__ == "__main__":
    main() 