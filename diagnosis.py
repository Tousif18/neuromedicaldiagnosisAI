import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import ml_models
import time
import base64

# CSS for diagnosis page
def diagnosis_css():
    return """
    <style>
    .diagnosis-container {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.8), rgba(22, 27, 34, 0.8));
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #FF00FF;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
        margin-bottom: 25px;
    }
    
    .cyberpunk-title {
        font-size: 2.5em;
        font-weight: 900;
        background: linear-gradient(to right, #FF00FF, #00FFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
    }
    
    .cyberpunk-subtitle {
        color: #00FFFF;
        font-size: 1.5em;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .feature-selection {
        background: rgba(0, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #00FFFF;
    }
    
    .diagnosis-card {
        background: rgba(13, 17, 23, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #00FFFF;
        transition: all 0.3s ease;
    }
    
    .diagnosis-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.3);
    }
    
    .section-title {
        color: #FF00FF;
        font-size: 1.2em;
        margin-bottom: 15px;
        border-bottom: 1px solid #FF00FF;
        padding-bottom: 5px;
    }
    
    .warning-box {
        background: rgba(255, 0, 0, 0.1);
        border-left: 3px solid #FF0000;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: rgba(0, 255, 0, 0.1);
        border-left: 3px solid #00FF00;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .info-box {
        background: rgba(0, 255, 255, 0.1);
        border-left: 3px solid #00FFFF;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .param-label {
        color: #FF00FF;
        font-weight: bold;
    }
    
    .result-box-positive {
        background: rgba(255, 0, 0, 0.1);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #FF5555;
        margin: 20px 0;
    }
    
    .result-box-negative {
        background: rgba(0, 255, 0, 0.1);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #55FF55;
        margin: 20px 0;
    }
    
    .scanner-container {
        position: relative;
        width: 100%;
        height: 5px;
        background: rgba(0, 255, 255, 0.1);
        margin: 20px 0;
        overflow: hidden;
    }
    
    .scanner-light {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, #00FFFF, transparent);
        animation: scan 2s linear infinite;
    }
    
    @keyframes scan {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
    
    .model-result {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        transition: all 0.3s;
    }
    
    .model-result:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .selector-container {
        background: rgba(255, 0, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #FF00FF;
    }
    
    .blinking-cursor {
        color: #00FFFF;
        font-weight: bold;
        animation: blink 1s step-start infinite;
    }
    
    @keyframes blink {
        50% {
            opacity: 0;
        }
    }
    
    .disease-selector {
        background: rgba(0, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #00FFFF;
    }
    
    .use-data-btn {
        background: linear-gradient(90deg, #FF00FF, #00FFFF);
        color: #0D1117;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
        display: block;
        width: 100%;
    }
    
    .use-data-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.3);
    }
    </style>
    """

def diagnosis_page():
    """Render the diagnosis page for making predictions with cyberpunk styling"""
    # Apply custom CSS
    st.markdown(diagnosis_css(), unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("Please login to access this page")
        return
    
    # Cyberpunk-styled title
    st.markdown('<h1 class="cyberpunk-title">Neural Diagnostic Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cyberpunk-subtitle">AI-Powered Medical Analysis System</p>', unsafe_allow_html=True)
    
    # Add futuristic scanner image
    st.image("assets/images/neural_scan.png", use_container_width=True)
    
    # Main diagnosis container
    st.markdown('<div class="diagnosis-container">', unsafe_allow_html=True)
    
    # Current time display with cyberpunk style
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<p style="text-align:right; color:#00FFFF; font-family:monospace; font-size:0.8em;">SYSTEM TIME: {current_time}</p>', unsafe_allow_html=True)
    
    # Get user data
    user_data = st.session_state.users[st.session_state.current_user]
    
    # Individual selector with cyberpunk styling
    st.markdown('<div class="selector-container">', unsafe_allow_html=True)
    st.markdown('<p style="color:#FF00FF; font-size:1.1em; margin-bottom:10px;">👤 Select Neural Profile for Diagnosis</p>', unsafe_allow_html=True)
    
    # Get list of individuals
    individuals = list(user_data["individuals"].keys())
    
    if not individuals:
        st.markdown(
            """
            <div class="warning-box">
                <span style="color:#FF0000; font-weight:bold;">⚠️ ERROR:</span> No neural profiles detected. Please add at least one individual in the Profile Vault.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)  # Close diagnosis container
        return
    
    # Show individual selection dropdown
    selected_individual = st.selectbox("", individuals)
    st.markdown('</div>', unsafe_allow_html=True)  # Close selector container
    
    # Disease selector with cyberpunk styling
    st.markdown('<div class="disease-selector">', unsafe_allow_html=True)
    st.markdown('<p style="color:#00FFFF; font-size:1.1em; margin-bottom:10px;">🔬 Select Target Disease</p>', unsafe_allow_html=True)
    
    # Get available diseases
    available_diseases = ml_models.get_available_diseases()
    
    # Show disease selection dropdown
    selected_disease = st.selectbox(
        "",
        available_diseases,
        index=0,
        help="Select the disease to diagnose"
    )
    
    # Set the current disease in the ML models
    ml_models.set_current_disease(selected_disease)
    st.markdown('</div>', unsafe_allow_html=True)  # Close disease selector container
    
    if selected_individual:
        individual_data = user_data["individuals"][selected_individual]
        
        # Diagnosis card
        st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
        st.markdown(f'<h3 class="section-title">🧬 Bio-Parameter Analysis for {selected_individual}</h3>', unsafe_allow_html=True)
        
        # Create a scanner animation
        st.markdown(
            """
            <div class="scanner-container">
                <div class="scanner-light"></div>
            </div>
            <p style="text-align:center; color:#00FFFF; font-family:monospace; font-size:0.9em; margin-bottom:20px;">
                <span class="blinking-cursor">_</span> ENTER BIOMETRIC DATA FOR AI ANALYSIS <span class="blinking-cursor">_</span>
            </p>
            """,
            unsafe_allow_html=True
        )
        
        # Get the required features for the selected disease
        required_features = ml_models.get_required_features(selected_disease)
        
        # Initialize form for manual entry
        with st.form("diagnosis_form"):
            # Split features into two columns
            mid_point = len(required_features) // 2
            left_features = required_features[:mid_point]
            right_features = required_features[mid_point:]
            
            col1, col2 = st.columns(2)
            
            # Dictionary to store input values
            input_data = {}
            
            with col1:
                for feature in left_features:
                    # Format feature name for display
                    display_name = " ".join([word.capitalize() for word in feature.split("_")])
                    
                    st.markdown(f'<p class="param-label">{display_name}</p>', unsafe_allow_html=True)
                    
                    # Determine appropriate input type
                    if feature.lower() in ['age', 'sex', 'gender', 'smoking', 'hypertension', 'heartdisease', 'familyhistory']:
                        # For binary or integer features
                        if feature.lower() in ['sex', 'gender', 'smoking', 'hypertension', 'heartdisease', 'familyhistory']:
                            help_text = "0 = No/Female, 1 = Yes/Male"
                            min_val = 0
                            max_val = 1
                        else:  # Age
                            help_text = "Enter age in years"
                            min_val = 0
                            max_val = 120
                        
                        value = st.number_input(
                            "",
                            min_value=min_val,
                            max_value=max_val,
                            key=f"left_{feature}",
                            help=help_text
                        )
                    else:
                        # For continuous features
                        value = st.number_input(
                            "",
                            min_value=0.0,
                            key=f"left_{feature}"
                        )
                    
                    input_data[feature] = value
            
            with col2:
                for feature in right_features:
                    # Format feature name for display
                    display_name = " ".join([word.capitalize() for word in feature.split("_")])
                    
                    st.markdown(f'<p class="param-label">{display_name}</p>', unsafe_allow_html=True)
                    
                    # Determine appropriate input type
                    if feature.lower() in ['age', 'sex', 'gender', 'smoking', 'hypertension', 'heartdisease', 'familyhistory']:
                        # For binary or integer features
                        if feature.lower() in ['sex', 'gender', 'smoking', 'hypertension', 'heartdisease', 'familyhistory']:
                            help_text = "0 = No/Female, 1 = Yes/Male"
                            min_val = 0
                            max_val = 1
                        else:  # Age
                            help_text = "Enter age in years"
                            min_val = 0
                            max_val = 120
                        
                        value = st.number_input(
                            "",
                            min_value=min_val,
                            max_value=max_val,
                            key=f"right_{feature}",
                            help=help_text
                        )
                    else:
                        # For continuous features
                        value = st.number_input(
                            "",
                            min_value=0.0,
                            key=f"right_{feature}"
                        )
                    
                    input_data[feature] = value
            
            # Submit button with custom styling
            submitted = st.form_submit_button("RUN NEURAL ANALYSIS")
            
            if submitted:
                # Check if at least some values are provided
                if all(v == 0 for v in input_data.values()):
                    st.markdown(
                        """
                        <div class="warning-box">
                            <span style="color:#FF0000; font-weight:bold;">⚠️ ERROR:</span> Please provide at least some parameter values for analysis.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Display prediction results
                    with st.spinner("Neural network processing..."):
                        # Simulate processing with a progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Run prediction with the selected disease
                        results = ml_models.run_prediction(input_data, selected_disease)
                        
                        # Remove progress bar
                        progress_bar.empty()
                        
                        # Display success message
                        st.markdown(
                            """
                            <div class="success-box">
                                <span style="color:#00FF00; font-weight:bold;">✅ SUCCESS:</span> Neural diagnostic analysis complete.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Check for error in results
                        if 'error' in results:
                            st.error(f"Error in analysis: {results['error']}")
                        else:
                            # Determine the overall prediction (majority vote)
                            predictions = [
                                results["svm_prediction"],
                                results["lr_prediction"],
                                results["rf_prediction"]
                            ]
                            overall_prediction = 1 if sum(predictions) >= 2 else 0
                            overall_confidence = (
                                results["svm_probability"] + 
                                results["lr_probability"] + 
                                results["rf_probability"]
                            ) / 3
                            
                            # Display overall result
                            result_class = "result-box-positive" if overall_prediction == 1 else "result-box-negative"
                            result_color = "#FF5555" if overall_prediction == 1 else "#55FF55"
                            result_text = "Positive (High Risk)" if overall_prediction == 1 else "Negative (Low Risk)"
                            
                            st.markdown(f"""
                            <div class="{result_class}">
                                <h3 style="color:{result_color}; margin-top:0;">AI Diagnosis Results</h3>
                                <p style="font-size:1.2em;">Disease: <strong>{selected_disease}</strong></p>
                                <p style="font-size:1.2em;">Analysis Result: <strong style="color:{result_color}">{result_text}</strong></p>
                                <p style="font-size:1.1em;">Confidence Level: <strong>{overall_confidence:.2%}</strong></p>
                                <p style="color:#00FFFF; font-family:monospace; margin-top:15px;">ANALYSIS TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display model details in expander
                            with st.expander("View Neural Model Results"):
                                st.markdown('<div style="background:rgba(13, 17, 23, 0.7); padding:15px; border-radius:8px; border:1px solid #00FFFF;">', unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                # Format model-specific results
                                models = {
                                    "Support Vector Machine": {
                                        "prediction": results["svm_prediction"],
                                        "probability": results["svm_probability"]
                                    },
                                    "Logistic Regression": {
                                        "prediction": results["lr_prediction"],
                                        "probability": results["lr_probability"]
                                    },
                                    "Random Forest": {
                                        "prediction": results["rf_prediction"],
                                        "probability": results["rf_probability"]
                                    }
                                }
                                
                                for i, (model_name, model_result) in enumerate(models.items()):
                                    with [col1, col2, col3][i]:
                                        pred = model_result["prediction"]
                                        prob = model_result["probability"]
                                        color = "#FF5555" if pred == 1 else "#55FF55"
                                        text = "Positive" if pred == 1 else "Negative"
                                        
                                        st.markdown(f"""
                                        <div class="model-result" style="border:1px solid {color}; background:rgba({255 if pred == 1 else 0}, {0 if pred == 1 else 255}, 0, 0.05);">
                                            <p style="font-weight:bold; margin-bottom:5px; color:#00FFFF;">{model_name}</p>
                                            <p style="color:{color}; font-weight:bold;">{text}</p>
                                            <p>Confidence: {prob:.2%}</p>
                                            <div style="height:5px; background:{color}; width:{prob*100}%; margin-top:5px; border-radius:3px;"></div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Store the diagnosis result in the individual's history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            diagnosis_record = {
                                "timestamp": timestamp,
                                "input_data": input_data,
                                "disease": selected_disease,
                                "overall_prediction": overall_prediction,
                                "overall_confidence": overall_confidence,
                                "model_results": {
                                    "svm": {
                                        "prediction": int(results["svm_prediction"]),
                                        "probability": float(results["svm_probability"])
                                    },
                                    "lr": {
                                        "prediction": int(results["lr_prediction"]),
                                        "probability": float(results["lr_probability"])
                                    },
                                    "rf": {
                                        "prediction": int(results["rf_prediction"]),
                                        "probability": float(results["rf_probability"])
                                    }
                                }
                            }
                            
                            user_data["diagnosis_history"][selected_individual].append(diagnosis_record)
                            
                            # Display disease-specific recommendations based on result
                            st.markdown('<h3 style="color:#00FFFF; margin-top:30px;">AI Medical Recommendations</h3>', unsafe_allow_html=True)
                            
                            if overall_prediction == 1:
                                st.markdown(f"""
                                <div style="background:rgba(255,0,0,0.1); padding:15px; border-radius:5px; border-left:5px solid #FF5555;">
                                    <h4 style="color:#FF5555; margin-top:0;">High Risk: {selected_disease}</h4>
                                    <ul>
                                        <li>Consult with a healthcare professional as soon as possible</li>
                                        <li>Schedule comprehensive screening for {selected_disease}</li>
                                        <li>Monitor related symptoms and vital signs regularly</li>
                                        <li>Maintain detailed health records and medication schedules</li>
                                        <li>Consider lifestyle adjustments recommended for {selected_disease} management</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background:rgba(0,255,0,0.1); padding:15px; border-radius:5px; border-left:5px solid #55FF55;">
                                    <h4 style="color:#55FF55; margin-top:0;">Low Risk: {selected_disease}</h4>
                                    <ul>
                                        <li>Continue regular health check-ups and screenings</li>
                                        <li>Maintain a healthy lifestyle with balanced nutrition</li>
                                        <li>Stay physically active and manage stress</li>
                                        <li>Monitor any changes in health indicators related to {selected_disease}</li>
                                        <li>Follow preventive care recommendations for your age and risk factors</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close diagnosis card
        
        # Option to use existing data from profile
        st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">💾 Use Stored Biometric Data</h3>', unsafe_allow_html=True)
        
        if "manual_entries" in individual_data["data"] and individual_data["data"]["manual_entries"]:
            entries = individual_data["data"]["manual_entries"]
            
            # Sort entries by timestamp (newest first)
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Create a list of entry labels with visual styling
            st.markdown('<div style="max-height:300px; overflow-y:auto; margin-bottom:20px;">', unsafe_allow_html=True)
            
            for i, entry in enumerate(entries):
                timestamp = entry['timestamp']
                glucose = entry.get('glucose', 'N/A')
                bmi = entry.get('bmi', 'N/A')
                age = entry.get('age', 'N/A')
                
                if st.button(f"Entry: {timestamp} | Glucose: {glucose} | BMI: {bmi} | Age: {age}", key=f"entry_{i}", help="Click to use this data for diagnosis"):
                    # Create input data dictionary from selected entry
                    # Map the entry data to the required features as best as possible
                    input_data = {}
                    
                    # Common mappings between database fields and disease features
                    field_mappings = {
                        'glucose': ['Glucose', 'BloodGlucoseRandom', 'AvgGlucose'],
                        'blood_pressure': ['BloodPressure', 'RestingBP'],
                        'skin_thickness': ['SkinThickness'],
                        'insulin': ['Insulin'],
                        'bmi': ['BMI'],
                        'diabetes_pedigree': ['DiabetesPedigreeFunction'],
                        'age': ['Age']
                    }
                    
                    # Fill in as many required features as possible
                    for db_field, possible_features in field_mappings.items():
                        if db_field in entry:
                            # Map this value to any matching required feature
                            for feature in possible_features:
                                if feature in required_features:
                                    input_data[feature] = entry[db_field]
                    
                    # For any missing required features, use 0
                    for feature in required_features:
                        if feature not in input_data:
                            input_data[feature] = 0
                    
                    # Display prediction results
                    with st.spinner("Neural network processing..."):
                        # Simulate processing with a progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Run prediction with the selected disease
                        results = ml_models.run_prediction(input_data, selected_disease)
                        
                        # Remove progress bar
                        progress_bar.empty()
                        
                        # Display success message
                        st.markdown(
                            """
                            <div class="success-box">
                                <span style="color:#00FF00; font-weight:bold;">✅ SUCCESS:</span> Neural diagnostic analysis complete.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Check for error in results
                        if 'error' in results:
                            st.error(f"Error in analysis: {results['error']}")
                        else:
                            # Determine the overall prediction (majority vote)
                            predictions = [
                                results["svm_prediction"],
                                results["lr_prediction"],
                                results["rf_prediction"]
                            ]
                            overall_prediction = 1 if sum(predictions) >= 2 else 0
                            overall_confidence = (
                                results["svm_probability"] + 
                                results["lr_probability"] + 
                                results["rf_probability"]
                            ) / 3
                            
                            # Display overall result
                            result_class = "result-box-positive" if overall_prediction == 1 else "result-box-negative"
                            result_color = "#FF5555" if overall_prediction == 1 else "#55FF55"
                            result_text = "Positive (High Risk)" if overall_prediction == 1 else "Negative (Low Risk)"
                            
                            st.markdown(f"""
                            <div class="{result_class}">
                                <h3 style="color:{result_color}; margin-top:0;">AI Diagnosis Results</h3>
                                <p style="font-size:1.2em;">Disease: <strong>{selected_disease}</strong></p>
                                <p style="font-size:1.2em;">Analysis Result: <strong style="color:{result_color}">{result_text}</strong></p>
                                <p style="font-size:1.1em;">Confidence Level: <strong>{overall_confidence:.2%}</strong></p>
                                <p style="color:#00FFFF; font-family:monospace; margin-top:15px;">ANALYSIS TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display model details in expander
                            with st.expander("View Neural Model Results"):
                                st.markdown('<div style="background:rgba(13, 17, 23, 0.7); padding:15px; border-radius:8px; border:1px solid #00FFFF;">', unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                # Format model-specific results
                                models = {
                                    "Support Vector Machine": {
                                        "prediction": results["svm_prediction"],
                                        "probability": results["svm_probability"]
                                    },
                                    "Logistic Regression": {
                                        "prediction": results["lr_prediction"],
                                        "probability": results["lr_probability"]
                                    },
                                    "Random Forest": {
                                        "prediction": results["rf_prediction"],
                                        "probability": results["rf_probability"]
                                    }
                                }
                                
                                for i, (model_name, model_result) in enumerate(models.items()):
                                    with [col1, col2, col3][i]:
                                        pred = model_result["prediction"]
                                        prob = model_result["probability"]
                                        color = "#FF5555" if pred == 1 else "#55FF55"
                                        text = "Positive" if pred == 1 else "Negative"
                                        
                                        st.markdown(f"""
                                        <div class="model-result" style="border:1px solid {color}; background:rgba({255 if pred == 1 else 0}, {0 if pred == 1 else 255}, 0, 0.05);">
                                            <p style="font-weight:bold; margin-bottom:5px; color:#00FFFF;">{model_name}</p>
                                            <p style="color:{color}; font-weight:bold;">{text}</p>
                                            <p>Confidence: {prob:.2%}</p>
                                            <div style="height:5px; background:{color}; width:{prob*100}%; margin-top:5px; border-radius:3px;"></div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Store the diagnosis result in the individual's history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            diagnosis_record = {
                                "timestamp": timestamp,
                                "input_data": input_data,
                                "disease": selected_disease,
                                "overall_prediction": overall_prediction,
                                "overall_confidence": overall_confidence,
                                "model_results": {
                                    "svm": {
                                        "prediction": int(results["svm_prediction"]),
                                        "probability": float(results["svm_probability"])
                                    },
                                    "lr": {
                                        "prediction": int(results["lr_prediction"]),
                                        "probability": float(results["lr_probability"])
                                    },
                                    "rf": {
                                        "prediction": int(results["rf_prediction"]),
                                        "probability": float(results["rf_probability"])
                                    }
                                }
                            }
                            
                            user_data["diagnosis_history"][selected_individual].append(diagnosis_record)
                            
                            # Display disease-specific recommendations based on result
                            st.markdown('<h3 style="color:#00FFFF; margin-top:30px;">AI Medical Recommendations</h3>', unsafe_allow_html=True)
                            
                            if overall_prediction == 1:
                                st.markdown(f"""
                                <div style="background:rgba(255,0,0,0.1); padding:15px; border-radius:5px; border-left:5px solid #FF5555;">
                                    <h4 style="color:#FF5555; margin-top:0;">High Risk: {selected_disease}</h4>
                                    <ul>
                                        <li>Consult with a healthcare professional as soon as possible</li>
                                        <li>Schedule comprehensive screening for {selected_disease}</li>
                                        <li>Monitor related symptoms and vital signs regularly</li>
                                        <li>Maintain detailed health records and medication schedules</li>
                                        <li>Consider lifestyle adjustments recommended for {selected_disease} management</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background:rgba(0,255,0,0.1); padding:15px; border-radius:5px; border-left:5px solid #55FF55;">
                                    <h4 style="color:#55FF55; margin-top:0;">Low Risk: {selected_disease}</h4>
                                    <ul>
                                        <li>Continue regular health check-ups and screenings</li>
                                        <li>Maintain a healthy lifestyle with balanced nutrition</li>
                                        <li>Stay physically active and manage stress</li>
                                        <li>Monitor any changes in health indicators related to {selected_disease}</li>
                                        <li>Follow preventive care recommendations for your age and risk factors</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close scrollable container
        else:
            st.markdown(
                """
                <div class="info-box">
                    <span style="color:#00FFFF; font-weight:bold;">ℹ️ INFORMATION:</span> No stored biometric data available for this neural profile. Add data in the Profile Vault.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close diagnosis card
    
    # Close the main diagnosis container
    st.markdown('</div>', unsafe_allow_html=True)

# CSS for history page
def history_css():
    """CSS for history page"""
    return """
    <style>
    .history-container {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.8), rgba(22, 27, 34, 0.8));
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #FF00FF;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
        margin-bottom: 25px;
    }
    
    .cyberpunk-title {
        font-size: 2.5em;
        font-weight: 900;
        background: linear-gradient(to right, #FF00FF, #00FFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
    }
    
    .cyberpunk-subtitle {
        color: #00FFFF;
        font-size: 1.5em;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .selector-container {
        background: rgba(255, 0, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #FF00FF;
    }
    
    .history-card {
        background: rgba(13, 17, 23, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #00FFFF;
    }
    
    .section-title {
        color: #FF00FF;
        font-size: 1.2em;
        margin-bottom: 15px;
        border-bottom: 1px solid #FF00FF;
        padding-bottom: 5px;
    }
    
    .warning-box {
        background: rgba(255, 0, 0, 0.1);
        border-left: 3px solid #FF0000;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .info-box {
        background: rgba(0, 255, 255, 0.1);
        border-left: 3px solid #00FFFF;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .timeline {
        position: relative;
        margin: 20px 0;
        padding-left: 30px;
    }
    
    .timeline:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 2px;
        background: linear-gradient(to bottom, #FF00FF, #00FFFF);
        border-radius: 2px;
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 25px;
        padding: 15px;
        background: rgba(13, 17, 23, 0.7);
        border-radius: 8px;
        border: 1px solid #00FFFF;
        transition: all 0.3s ease;
    }
    
    .timeline-item:hover {
        transform: translateX(5px);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    .timeline-item:before {
        content: "";
        position: absolute;
        left: -34px;
        top: 15px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #FF00FF;
        border: 2px solid #00FFFF;
        box-shadow: 0 0 5px #00FFFF;
    }
    
    .timeline-date {
        color: #00FFFF;
        font-family: monospace;
        font-size: 0.8em;
        margin-bottom: 8px;
    }
    
    .timeline-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .timeline-disease {
        color: #FF00FF;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .timeline-result {
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9em;
    }
    
    .timeline-details {
        margin-top: 10px;
        padding: 10px;
        background: rgba(13, 17, 23, 0.5);
        border-radius: 5px;
        font-size: 0.9em;
    }
    
    .positive-tag {
        background-color: rgba(255, 0, 0, 0.2);
        color: #FF5555;
        border: 1px solid #FF5555;
    }
    
    .negative-tag {
        background-color: rgba(0, 255, 0, 0.2);
        color: #55FF55;
        border: 1px solid #55FF55;
    }
    
    .download-btn {
        background: linear-gradient(90deg, #FF00FF, #00FFFF);
        color: #0D1117;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
        display: block;
        text-decoration: none;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.3);
    }
    
    .confidence-bar {
        height: 5px;
        margin-top: 8px;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .confidence-value {
        height: 100%;
        border-radius: 3px;
    }
    </style>
    """

def history_page():
    """Render the diagnosis history page with cyberpunk styling"""
    # Apply custom CSS
    st.markdown(history_css(), unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("Please login to access this page")
        return
    
    # Cyberpunk-styled title
    st.markdown('<h1 class="cyberpunk-title">Neural Analysis Archives</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cyberpunk-subtitle">Historical Medical Diagnostic Records</p>', unsafe_allow_html=True)
    
    # Add futuristic banner image
    st.image("assets/images/history_timeline.png", use_container_width=True)
    
    # Main history container
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    
    # Current time display with cyberpunk style
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<p style="text-align:right; color:#00FFFF; font-family:monospace; font-size:0.8em;">SYSTEM TIME: {current_time}</p>', unsafe_allow_html=True)
    
    # Get user data
    user_data = st.session_state.users[st.session_state.current_user]
    
    # Individual selector with cyberpunk styling
    st.markdown('<div class="selector-container">', unsafe_allow_html=True)
    st.markdown('<p style="color:#FF00FF; font-size:1.1em; margin-bottom:10px;">👤 Access Neural Profile Archives</p>', unsafe_allow_html=True)
    
    # Get list of individuals
    individuals = list(user_data["individuals"].keys())
    
    if not individuals:
        st.markdown(
            """
            <div class="warning-box">
                <span style="color:#FF0000; font-weight:bold;">⚠️ ERROR:</span> No neural profiles detected. Please add at least one individual in the Profile Vault.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)  # Close history container
        return
    
    # Show individual selection dropdown
    selected_individual = st.selectbox("", individuals)
    st.markdown('</div>', unsafe_allow_html=True)  # Close selector container
    
    if selected_individual:
        # History card
        st.markdown('<div class="history-card">', unsafe_allow_html=True)
        
        # Get diagnosis history for the selected individual
        history = user_data["diagnosis_history"].get(selected_individual, [])
        
        if not history:
            st.markdown(
                f"""
                <div class="info-box">
                    <span style="color:#00FFFF; font-weight:bold;">ℹ️ INFORMATION:</span> No diagnostic records found for {selected_individual}. Run a neural analysis first.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Sort history by timestamp (newest first)
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Add export option
            if len(history) > 0:
                st.markdown('<div style="text-align:right; margin-bottom:20px;">', unsafe_allow_html=True)
                
                # Create CSV data
                csv_data = ""
                
                # Define headers
                headers = ["Timestamp", "Disease", "Result", "Confidence", "SVM_Pred", "SVM_Conf", "LR_Pred", "LR_Conf", "RF_Pred", "RF_Conf"]
                
                # Add input data headers (we'll take from first record)
                for key in history[0]["input_data"].keys():
                    formatted_key = " ".join([word.capitalize() for word in key.split("_")])
                    headers.append(f"Input_{formatted_key}")
                
                csv_data += ",".join(headers) + "\n"
                
                # Add data rows
                for record in history:
                    # Format basic data
                    row = [
                        record["timestamp"],
                        record["disease"],
                        "Positive" if record["overall_prediction"] == 1 else "Negative",
                        str(record["overall_confidence"]),
                        "Positive" if record["model_results"]["svm"]["prediction"] == 1 else "Negative",
                        str(record["model_results"]["svm"]["probability"]),
                        "Positive" if record["model_results"]["lr"]["prediction"] == 1 else "Negative",
                        str(record["model_results"]["lr"]["probability"]),
                        "Positive" if record["model_results"]["rf"]["prediction"] == 1 else "Negative",
                        str(record["model_results"]["rf"]["probability"])
                    ]
                    
                    # Add input data values
                    for key in history[0]["input_data"].keys():
                        value = record["input_data"].get(key, "")
                        row.append(str(value))
                    
                    csv_data += ",".join(row) + "\n"
                
                # Convert to downloadable link
                b64 = base64.b64encode(csv_data.encode()).decode()
                download_link = f'<a href="data:file/csv;base64,{b64}" download="{selected_individual}_diagnosis_history.csv" class="download-btn">📊 EXPORT DIAGNOSTIC ARCHIVE</a>'
                st.markdown(download_link, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display timeline of diagnoses
            st.markdown('<h3 class="section-title">📈 Neural Diagnostic Timeline</h3>', unsafe_allow_html=True)
            st.markdown('<div class="timeline">', unsafe_allow_html=True)
            
            # Display each diagnosis record in timeline format
            for record in history:
                # Format timestamp
                timestamp = record["timestamp"]
                
                # Format result
                result_class = "positive-tag" if record["overall_prediction"] == 1 else "negative-tag"
                result_text = "Positive (High Risk)" if record["overall_prediction"] == 1 else "Negative (Low Risk)"
                result_color = "#FF5555" if record["overall_prediction"] == 1 else "#55FF55"
                
                # Format confidence bar
                confidence = record["overall_confidence"]
                
                # Disease name
                disease = record["disease"]
                
                st.markdown(
                    f"""
                    <div class="timeline-item">
                        <div class="timeline-date">{timestamp}</div>
                        <div class="timeline-title">
                            <span class="timeline-disease">{disease}</span>
                            <span class="timeline-result {result_class}">{result_text}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-value" style="width:{confidence*100}%; background:{result_color};"></div>
                        </div>
                        <p style="margin-top:5px; font-size:0.9em;">Confidence: {confidence:.2%}</p>
                    """,
                    unsafe_allow_html=True
                )
                
                # Create expandable details
                with st.expander("View detailed results"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**Input Parameters:**")
                        for key, value in record["input_data"].items():
                            # Format key for display
                            display_key = " ".join([word.capitalize() for word in key.split("_")])
                            st.markdown(f"- {display_key}: {value}")
                    
                    with col2:
                        # Model results
                        st.markdown("**AI Model Results:**")
                        
                        models_col1, models_col2, models_col3 = st.columns(3)
                        
                        # SVM
                        with models_col1:
                            svm_pred = record["model_results"]["svm"]["prediction"]
                            svm_prob = record["model_results"]["svm"]["probability"]
                            svm_color = "#FF5555" if svm_pred == 1 else "#55FF55"
                            svm_text = "Positive" if svm_pred == 1 else "Negative"
                            
                            st.markdown(f"""
                            <div style="padding:10px; border-radius:5px; border:1px solid {svm_color}; background:rgba({255 if svm_pred == 1 else 0}, {0 if svm_pred == 1 else 255}, 0, 0.05);">
                                <p style="font-weight:bold; margin-bottom:5px; color:#00FFFF;">SVM</p>
                                <p style="color:{svm_color}; font-weight:bold;">{svm_text}</p>
                                <p>Confidence: {svm_prob:.2%}</p>
                                <div style="height:5px; background:{svm_color}; width:{svm_prob*100}%; margin-top:5px; border-radius:3px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Logistic Regression
                        with models_col2:
                            lr_pred = record["model_results"]["lr"]["prediction"]
                            lr_prob = record["model_results"]["lr"]["probability"]
                            lr_color = "#FF5555" if lr_pred == 1 else "#55FF55"
                            lr_text = "Positive" if lr_pred == 1 else "Negative"
                            
                            st.markdown(f"""
                            <div style="padding:10px; border-radius:5px; border:1px solid {lr_color}; background:rgba({255 if lr_pred == 1 else 0}, {0 if lr_pred == 1 else 255}, 0, 0.05);">
                                <p style="font-weight:bold; margin-bottom:5px; color:#00FFFF;">Logistic Regression</p>
                                <p style="color:{lr_color}; font-weight:bold;">{lr_text}</p>
                                <p>Confidence: {lr_prob:.2%}</p>
                                <div style="height:5px; background:{lr_color}; width:{lr_prob*100}%; margin-top:5px; border-radius:3px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Random Forest
                        with models_col3:
                            rf_pred = record["model_results"]["rf"]["prediction"]
                            rf_prob = record["model_results"]["rf"]["probability"]
                            rf_color = "#FF5555" if rf_pred == 1 else "#55FF55"
                            rf_text = "Positive" if rf_pred == 1 else "Negative"
                            
                            st.markdown(f"""
                            <div style="padding:10px; border-radius:5px; border:1px solid {rf_color}; background:rgba({255 if rf_pred == 1 else 0}, {0 if rf_pred == 1 else 255}, 0, 0.05);">
                                <p style="font-weight:bold; margin-bottom:5px; color:#00FFFF;">Random Forest</p>
                                <p style="color:{rf_color}; font-weight:bold;">{rf_text}</p>
                                <p>Confidence: {rf_prob:.2%}</p>
                                <div style="height:5px; background:{rf_color}; width:{rf_prob*100}%; margin-top:5px; border-radius:3px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Close timeline item div
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Close timeline div
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add analytics section
            if len(history) > 1:
                st.markdown('<h3 class="section-title">📊 Diagnostic Analytics</h3>', unsafe_allow_html=True)
                
                # Count diseases and results
                disease_counts = {}
                positive_counts = {}
                
                for record in history:
                    disease = record["disease"]
                    result = record["overall_prediction"]
                    
                    if disease not in disease_counts:
                        disease_counts[disease] = 0
                        positive_counts[disease] = 0
                    
                    disease_counts[disease] += 1
                    if result == 1:
                        positive_counts[disease] += 1
                
                # Display disease statistics
                chart_data = []
                for disease, count in disease_counts.items():
                    positive = positive_counts[disease]
                    negative = count - positive
                    chart_data.append({
                        "Disease": disease,
                        "Positive": positive,
                        "Negative": negative
                    })
                
                chart_df = pd.DataFrame(chart_data)
                
                # Use Streamlit's built-in chart for this visualization
                st.bar_chart(chart_df.set_index("Disease"))
                
                # Display date-based trends if enough data is available
                if len(history) >= 3:
                    st.markdown('<h4 style="color:#00FFFF; margin-top:20px;">Temporal Trends</h4>', unsafe_allow_html=True)
                    
                    # Sort by timestamp (oldest first)
                    history_sorted = sorted(history, key=lambda x: x.get("timestamp", ""))
                    
                    # For each disease, plot the confidence trend over time
                    for disease in disease_counts.keys():
                        disease_history = [record for record in history_sorted if record["disease"] == disease]
                        
                        if len(disease_history) >= 2:
                            confidence_data = []
                            for record in disease_history:
                                timestamp = record["timestamp"]
                                confidence = record["overall_confidence"]
                                confidence_data.append({
                                    "Timestamp": timestamp,
                                    "Confidence": confidence
                                })
                            
                            confidence_df = pd.DataFrame(confidence_data)
                            
                            # Display trend for this disease
                            st.markdown(f'<p style="color:#FF00FF; margin-top:15px;">{disease} Confidence Trend:</p>', unsafe_allow_html=True)
                            st.line_chart(confidence_df.set_index("Timestamp"))
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close history card
    
    # Close the main history container
    st.markdown('</div>', unsafe_allow_html=True)