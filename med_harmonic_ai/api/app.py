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
    pages = ["Getting Started", "Home", "Predict Compound Efficacy", "Multi-System Integration", "About"]
    
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
        ### {tooltip("System-Specific Properties", "These properties are specific to the medical system you selected and influence how the compound works")}
        """, unsafe_allow_html=True)
        
        # Create different inputs based on selected system
        if medical_system == "Ayurvedic":
            st.markdown("""<div class="explanation-box">
            <p><b>Ayurvedic medicine</b> is based on traditional Indian practices that focus on balancing bodily systems through natural remedies.</p>
            </div>""", unsafe_allow_html=True)
            
            is_bitter = st.checkbox("Is Bitter", 
                                   help="Bitter taste (rasa) often indicates compounds that can lower blood sugar")
            is_cooling = st.checkbox("Is Cooling", 
                                    help="Cooling properties (virya) can help reduce inflammation associated with diabetes")
            reduces_kapha = st.checkbox("Reduces Kapha", 
                                       help="Kapha reduction helps with metabolic balance, important for diabetes")
            # Default allopathic properties
            half_life_hours = 0.0
            bioavailability = 0.0
            # Default unani properties
            is_hot = False
            reduces_phlegm = False
            # Default homeopathic properties
            has_thirst_symptom = False
            has_urination_symptom = False
            
        elif medical_system == "Allopathic":
            st.markdown("""<div class="explanation-box">
            <p><b>Allopathic medicine</b> is conventional Western medicine based on scientific research and evidence-based practices.</p>
            </div>""", unsafe_allow_html=True)
            
            # Default ayurvedic properties
            is_bitter = False
            is_cooling = False
            reduces_kapha = False
            # Allopathic properties
            half_life_hours = st.slider("Half-life (hours)", 0.0, 24.0, 6.0, 
                                       help="The time it takes for half of the drug to be eliminated from the body")
            bioavailability = st.slider("Bioavailability", 0.0, 1.0, 0.5, 
                                       help="The fraction of the drug that reaches systemic circulation")
            # Default unani properties
            is_hot = False
            reduces_phlegm = False
            # Default homeopathic properties
            has_thirst_symptom = False
            has_urination_symptom = False
            
        elif medical_system == "Unani":
            st.markdown("""<div class="explanation-box">
            <p><b>Unani medicine</b> is a Greco-Arabic medical tradition that uses natural remedies to balance the body's four humors.</p>
            </div>""", unsafe_allow_html=True)
            
            # Default ayurvedic properties
            is_bitter = False
            is_cooling = False
            reduces_kapha = False
            # Default allopathic properties
            half_life_hours = 0.0
            bioavailability = 0.0
            # Unani properties
            is_hot = st.checkbox("Is Hot", 
                                help="Hot temperament (mizaj) can increase metabolic activity")
            reduces_phlegm = st.checkbox("Reduces Phlegm", 
                                        help="Phlegm reduction can improve metabolic function relevant to diabetes")
            # Default homeopathic properties
            has_thirst_symptom = False
            has_urination_symptom = False
            
        elif medical_system == "Homeopathic":
            st.markdown("""<div class="explanation-box">
            <p><b>Homeopathic medicine</b> follows the principle that 'like cures like' using highly diluted substances to stimulate healing.</p>
            </div>""", unsafe_allow_html=True)
            
            # Default ayurvedic properties
            is_bitter = False
            is_cooling = False
            reduces_kapha = False
            # Default allopathic properties
            half_life_hours = 0.0
            bioavailability = 0.0
            # Default unani properties
            is_hot = False
            reduces_phlegm = False
            # Homeopathic properties
            has_thirst_symptom = st.checkbox("Addresses Thirst Symptoms", 
                                            help="Targets excessive thirst, a common diabetes symptom")
            has_urination_symptom = st.checkbox("Addresses Urination Symptoms", 
                                               help="Targets frequent urination, a common diabetes symptom")
    
    # Common properties
    with col2:
        st.markdown(f"""
        ### {tooltip("Common Properties", "These properties apply to compounds from all medical systems")}
        """, unsafe_allow_html=True)
        
        active_compounds_count = st.slider("Number of Active Compounds", 1, 10, 3, 
                                          help="How many bioactive components are in this compound")
        has_blood_sugar_effect = st.checkbox("Has Blood Sugar Effect", 
                                           help="Check if the compound is known to affect blood sugar levels")
        
        st.markdown(f"""
        ### {tooltip("Harmonic Properties", "These represent how the compound interacts with the body's systems and other compounds")}
        """, unsafe_allow_html=True)
        
        # Pre-set system frequency based on selection
        system_frequency = get_system_frequency(medical_system)
        st.markdown(f"""
        System Frequency: {system_frequency}
        <span style="font-size: 0.8em; color: #666;">(Pre-set based on the medical system)</span>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        {tooltip("Efficacy Amplitude", "How strongly the compound affects the target condition (diabetes)")}
        """, unsafe_allow_html=True)
        efficacy_amplitude = st.slider("Efficacy Amplitude", 0.0, 1.0, 0.7, 
                                     help="Higher values indicate stronger effects")
        
        st.markdown(f"""
        {tooltip("Phase Alignment", "How well the compound's effects synchronize with the body's natural processes")}
        """, unsafe_allow_html=True)
        phase_alignment = st.slider("Phase Alignment", 0.0, 1.0, 0.5, 
                                  help="Higher values indicate better integration with the body's systems")
        
        st.markdown(f"""
        {tooltip("Resonance Potential", "How well this compound can enhance or work with other compounds")}
        """, unsafe_allow_html=True)
        resonance_potential = st.slider("Resonance Potential", 0.0, 1.0, 0.5, 
                                      help="Higher values indicate better synergy with other treatments")
    
    # Create compound data dictionary
    compound_data = {
        # System-specific properties
        'is_bitter': int(is_bitter),
        'is_cooling': int(is_cooling),
        'reduces_kapha': int(reduces_kapha),
        'half_life_hours': half_life_hours,
        'bioavailability': bioavailability,
        'is_hot': int(is_hot),
        'reduces_phlegm': int(reduces_phlegm),
        'has_thirst_symptom': int(has_thirst_symptom),
        'has_urination_symptom': int(has_urination_symptom),
        # Common properties
        'active_compounds_count': active_compounds_count,
        'has_blood_sugar_effect': int(has_blood_sugar_effect),
        # Harmonic properties
        'system_frequency': system_frequency,
        'efficacy_amplitude': efficacy_amplitude,
        'phase_alignment': phase_alignment,
        'resonance_potential': resonance_potential
    }
    
    # Predict button with clear explanation
    st.markdown("""
    <div class="explanation-box">
    <p>Click the button below to have the AI predict whether this compound would be effective for diabetes treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Predict Efficacy", help="Run the AI prediction"):
        st.markdown("### Prediction Results")
        
        try:
            # Add a spinner to show processing
            with st.spinner('Analyzing compound properties...'):
                # Run prediction
                prediction = predictor.predict_single_compound(compound_data)
            
            # Display results
            if prediction['is_effective']:
                st.success(f"✅ **{compound_name}** is predicted to be **EFFECTIVE** for diabetes treatment.")
                st.info(f"Confidence: {prediction['confidence']:.2%}")
                st.markdown("""
                <div class="explanation-box">
                <p>The model predicts this compound has properties that would help treat diabetes. The confidence percentage indicates how certain the AI is about this prediction.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ **{compound_name}** is predicted to be **NOT EFFECTIVE** for diabetes treatment.")
                st.info(f"Confidence: {(1 - prediction['confidence']):.2%}")
                st.markdown("""
                <div class="explanation-box">
                <p>The model predicts this compound would not be effective for diabetes treatment. Try adjusting some properties to see what might make it more effective.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualize some properties
            st.markdown("### Compound Properties Visualization")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            properties = ['Efficacy', 'Safety', 'Sustainability']
            values = [efficacy_amplitude, 0.95 - resonance_potential * 0.3, phase_alignment]
            
            ax.bar(properties, values, color=['#1f77b4', '#2ca02c', '#d62728'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title(f'Predicted Properties for {compound_name}')
            
            st.pyplot(fig)
            
            # Add suggested improvements
            st.markdown("### Suggested Improvements")
            
            if not prediction['is_effective']:
                improvements = []
                
                if has_blood_sugar_effect == 0:
                    improvements.append("- Add components with direct blood sugar effects")
                
                if efficacy_amplitude < 0.6:
                    improvements.append("- Increase the efficacy amplitude (potency)")
                
                if phase_alignment < 0.5:
                    improvements.append("- Improve phase alignment with body processes")
                
                if medical_system == "Ayurvedic" and not reduces_kapha:
                    improvements.append("- Include ingredients that reduce Kapha")
                    
                if medical_system == "Homeopathic" and not (has_thirst_symptom or has_urination_symptom):
                    improvements.append("- Choose remedies that address key diabetes symptoms")
                
                if improvements:
                    st.markdown("Try these changes to potentially improve effectiveness:")
                    for improvement in improvements:
                        st.markdown(improvement)
                else:
                    st.markdown("Try increasing the values of multiple properties slightly to improve effectiveness.")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("If you're seeing feature name errors, ensure the model was trained in Python 3.9 environment.")

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

if __name__ == "__main__":
    main() 