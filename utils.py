import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime

def get_current_timestamp():
    """Get current timestamp in string format
    
    Returns:
        str: Current timestamp in YYYY-MM-DD HH:MM:SS format
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_individual_names(user_data):
    """Get list of individual names for the current user
    
    Args:
        user_data (dict): Current user's data
        
    Returns:
        list: List of individual names
    """
    if "individuals" in user_data:
        return list(user_data["individuals"].keys())
    return []

def get_latest_medical_data(individual_data):
    """Get the latest medical data for an individual
    
    Args:
        individual_data (dict): Individual's data
        
    Returns:
        dict: Latest medical data or empty dict if none available
    """
    if "manual_entries" in individual_data["data"] and individual_data["data"]["manual_entries"]:
        return individual_data["data"]["manual_entries"][-1]
    return {}

def validate_input_data(input_data):
    """Validate input data for diagnosis
    
    Args:
        input_data (dict): Input data for diagnosis
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    
    for field in required_fields:
        if field not in input_data or input_data[field] == 0:
            return False, f"{field} is required and must be greater than 0."
    
    return True, ""

def format_diagnosis_result(results):
    """Format diagnosis results for display
    
    Args:
        results (dict): Diagnosis results from ML models
        
    Returns:
        dict: Formatted results for display
    """
    formatted = {}
    
    # Format model results
    for model in ['svm', 'lr', 'rf']:
        prediction = results[f'{model}_prediction']
        probability = results[f'{model}_probability']
        
        formatted[model] = {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'probability': f"{probability:.2%}",
            'is_positive': prediction == 1
        }
    
    # Calculate overall assessment
    positive_count = sum([
        results['svm_prediction'],
        results['lr_prediction'],
        results['rf_prediction']
    ])
    
    formatted['overall'] = {
        'positive_count': positive_count,
        'is_majority_positive': positive_count >= 2,
        'assessment': 'Majority of models indicate a positive result.' if positive_count >= 2 else 'Majority of models indicate a negative result.'
    }
    
    return formatted

def export_diagnosis_history(history_records):
    """Export diagnosis history to CSV
    
    Args:
        history_records (list): List of diagnosis records
        
    Returns:
        str: Base64 encoded CSV data
    """
    # Prepare data for CSV
    data = []
    
    for record in history_records:
        row = {
            'Timestamp': record['timestamp'],
            'Glucose': record['input_data']['Glucose'],
            'BloodPressure': record['input_data']['BloodPressure'],
            'SkinThickness': record['input_data']['SkinThickness'],
            'Insulin': record['input_data']['Insulin'],
            'BMI': record['input_data']['BMI'],
            'DiabetesPedigreeFunction': record['input_data']['DiabetesPedigreeFunction'],
            'Age': record['input_data']['Age'],
            'SVM_Prediction': 'Positive' if record['results']['svm_prediction'] == 1 else 'Negative',
            'SVM_Probability': f"{record['results']['svm_probability']:.2%}",
            'LR_Prediction': 'Positive' if record['results']['lr_prediction'] == 1 else 'Negative',
            'LR_Probability': f"{record['results']['lr_probability']:.2%}",
            'RF_Prediction': 'Positive' if record['results']['rf_prediction'] == 1 else 'Negative',
            'RF_Probability': f"{record['results']['rf_probability']:.2%}"
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Encode as base64
    b64 = base64.b64encode(csv.encode()).decode()
    
    return b64
