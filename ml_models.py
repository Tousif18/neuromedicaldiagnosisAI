import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import json
import database

# Disease datasets for training models
DISEASE_DATASETS = {
    'Diabetes': {
        'features': [
            'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
            'BMI', 'DiabetesPedigreeFunction', 'Age'
        ],
        'target': 'Outcome',
        'data': {
            'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 110, 168, 139, 189, 166],
            'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96, 74, 88, 80, 60, 72],
            'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0, 26, 42, 0, 23, 19],
            'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0, 0, 170, 0, 846, 175],
            'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31, 35.3, 30.5, 0, 30, 38.2, 43.1, 30.1, 25.8],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232, 0.5, 0.42, 1.222, 0.398, 0.587],
            'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 59, 62, 41, 23, 51],
            'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
        },
        'description': 'Diabetes prediction based on diagnostic measurements'
    },
    'Heart Disease': {
        'features': [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ],
        'target': 'HeartDisease',
        'data': {
            'Age': [40, 49, 37, 48, 54, 39, 45, 54, 37, 48, 51, 52, 58, 44, 56],
            'Sex': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            'ChestPainType': [1, 2, 0, 2, 1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0],
            'RestingBP': [140, 160, 130, 138, 150, 120, 130, 125, 140, 130, 125, 140, 136, 120, 140],
            'Cholesterol': [289, 180, 283, 214, 195, 339, 237, 224, 203, 275, 213, 221, 205, 263, 294],
            'FastingBS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'RestingECG': [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            'MaxHR': [172, 156, 98, 108, 122, 170, 170, 155, 170, 139, 166, 178, 122, 178, 153],
            'ExerciseAngina': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'Oldpeak': [0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0.2, 0, 0, 1.5, 0, 1.3],
            'ST_Slope': [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            'HeartDisease': [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
        },
        'description': 'Heart disease prediction based on various clinical parameters'
    },
    'Stroke': {
        'features': [
            'Age', 'Sex', 'Hypertension', 'HeartDisease', 'AvgGlucose', 
            'BMI', 'Smoking'
        ],
        'target': 'Stroke',
        'data': {
            'Age': [67, 61, 80, 49, 79, 81, 74, 69, 59, 78, 45, 52, 70, 68, 63],
            'Sex': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Hypertension': [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'HeartDisease': [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'AvgGlucose': [228.69, 202.21, 105.92, 171.23, 174.12, 186.21, 70.09, 94.39, 76.15, 58.57, 80.43, 97.92, 103.63, 104.39, 85.28],
            'BMI': [36.6, 28.5, 32.5, 34.4, 24, 29, 27.4, 22.8, 29, 24.2, 26.1, 22.2, 35.5, 36.6, 25.7],
            'Smoking': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'Stroke': [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
        },
        'description': 'Stroke prediction based on health parameters'
    },
    'Liver Disease': {
        'features': [
            'Age', 'Gender', 'TotalBilirubin', 'DirectBilirubin', 
            'AlkalinePhosphatase', 'AlanineTransaminase', 'AspartateTransaminase', 
            'TotalProteins', 'Albumin', 'AlbuminGlobulinRatio'
        ],
        'target': 'LiverDisease',
        'data': {
            'Age': [65, 62, 62, 58, 72, 46, 26, 29, 17, 55, 57, 72, 64, 52, 50],
            'Gender': [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            'TotalBilirubin': [0.7, 10.9, 7.3, 1, 3.9, 0.9, 0.9, 0.9, 0.9, 7.6, 0.6, 7.1, 1.8, 3.6, 1],
            'DirectBilirubin': [0.1, 5.5, 4.1, 0.3, 1.9, 0.2, 0.3, 0.3, 0.3, 3, 0.1, 3.5, 0.8, 0.8, 0.2],
            'AlkalinePhosphatase': [187, 699, 490, 242, 195, 97, 182, 279, 248, 544, 289, 328, 263, 222, 259],
            'AlanineTransaminase': [16, 64, 60, 50, 27, 24, 35, 72, 131, 57, 21, 31, 49, 19, 140],
            'AspartateTransaminase': [18, 100, 68, 78, 59, 36, 31, 44, 90, 147, 23, 74, 17, 56, 51],
            'TotalProteins': [6.8, 7.5, 7, 7, 7.3, 7, 7.4, 7.4, 7.1, 6.7, 5.8, 7.1, 5.9, 7.5, 5.8],
            'Albumin': [3.3, 3.2, 3.5, 3.4, 2.4, 3.7, 4.1, 4.2, 4.2, 2.2, 2.6, 3.2, 3, 3.4, 2.7],
            'AlbuminGlobulinRatio': [0.9, 0.74, 1, 0.9, 0.5, 1.2, 1.2, 1.32, 1.45, 0.5, 0.8, 0.8, 1, 0.8, 0.87],
            'LiverDisease': [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
        },
        'description': 'Liver disease prediction based on patient data'
    },
    'Kidney Disease': {
        'features': [
            'Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar', 
            'RedBloodCells', 'PusCells', 'PusCellClumps', 'Bacteria', 'BloodGlucoseRandom', 
            'BloodUrea', 'SerumCreatinine', 'Sodium', 'Potassium', 'Hemoglobin', 
            'PackedCellVolume', 'WhiteBloodCellCount', 'RedBloodCellCount'
        ],
        'target': 'KidneyDisease',
        'data': {
            'Age': [48, 53, 63, 68, 61, 55, 80, 68, 53, 60, 48, 59, 63, 68, 80],
            'BloodPressure': [80, 70, 80, 80, 80, 80, 80, 70, 80, 90, 70, 70, 100, 70, 80],
            'SpecificGravity': [1.02, 1.01, 1.01, 1.01, 1.015, 1.02, 1.025, 1.01, 1.02, 1.01, 1.005, 1.01, 1.01, 1.015, 1.02],
            'Albumin': [1, 4, 2, 4, 2, 0, 0, 3, 1, 4, 4, 0, 2, 3, 1],
            'Sugar': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'RedBloodCells': [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            'PusCells': [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
            'PusCellClumps': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'Bacteria': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'BloodGlucoseRandom': [121, 160, 140, 131, 173, 140, 70, 133, 76, 148, 142, 64, 59, 149, 74],
            'BloodUrea': [36, 90, 70, 142, 148, 39, 49, 40, 25, 72, 60, 26, 47, 120, 82],
            'SerumCreatinine': [1.2, 2.4, 1.4, 4.2, 3.9, 1, 1.1, 1.2, 1.3, 5.1, 1.9, 0.9, 1.4, 5.6, 3.5],
            'Sodium': [140, 139, 150, 136, 135, 141, 143, 132, 129, 131, 137, 136, 142, 135, 133],
            'Potassium': [4.7, 5.2, 5.2, 4.7, 5.2, 4.1, 5.4, 4.1, 4, 5.5, 4.1, 4.5, 4.9, 5.5, 5.5],
            'Hemoglobin': [15.4, 9.4, 10.1, 7.1, 11, 11.2, 12, 10.2, 16.2, 10.1, 12.5, 17.8, 13.2, 7.7, 13],
            'PackedCellVolume': [44, 31, 32, 23, 32, 32, 41, 33, 47, 30, 38, 53, 41, 24, 40],
            'WhiteBloodCellCount': [7800, 8900, 6900, 9600, 5900, 6000, 4500, 7500, 6700, 9000, 8300, 6000, 5800, 9800, 9400],
            'RedBloodCellCount': [5.2, 3.7, 3.9, 3.2, 3.7, 3.7, 4.9, 4.5, 5.5, 3.6, 4.5, 6.6, 5, 3.4, 4.9],
            'KidneyDisease': [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
        },
        'description': 'Kidney disease prediction based on clinical parameters'
    }
}

# Additional diseases to support (more basic feature sets)
ADDITIONAL_DISEASES = [
    'Hypertension', 'Asthma', 'Arthritis', 'Migraine', 'Cancer',
    'Pneumonia', 'Tuberculosis', 'Alzheimer\'s', 'Parkinson\'s', 
    'Thyroid Disease', 'Osteoporosis', 'Anemia', 'Dementia', 'Hepatitis', 
    'COPD'
]

def initialize_additional_disease_datasets():
    """Generate basic datasets for additional diseases"""
    for disease in ADDITIONAL_DISEASES:
        if disease not in DISEASE_DATASETS:
            # Create a basic dataset with common health metrics
            DISEASE_DATASETS[disease] = {
                'features': [
                    'Age', 'Sex', 'BloodPressure', 'HeartRate', 
                    'RespiratoryRate', 'Temperature', 'OxygenSaturation',
                    'Weight', 'Height', 'BMI', 'FamilyHistory'
                ],
                'target': 'Outcome',
                'data': generate_synthetic_health_data(disease, 20),
                'description': f'{disease} prediction based on general health parameters'
            }

def generate_synthetic_health_data(disease_name, n_samples=20):
    """Generate synthetic health data for a disease
    
    Args:
        disease_name (str): The name of the disease
        n_samples (int): Number of samples to generate
        
    Returns:
        dict: Dictionary with feature data
    """
    np.random.seed(hash(disease_name) % 10000)  # Use disease name as seed for reproducibility
    
    # Generate data based on general health parameters
    data = {
        'Age': np.random.randint(18, 85, n_samples).tolist(),
        'Sex': np.random.randint(0, 2, n_samples).tolist(),
        'BloodPressure': np.random.randint(90, 180, n_samples).tolist(),
        'HeartRate': np.random.randint(60, 120, n_samples).tolist(),
        'RespiratoryRate': np.random.randint(12, 25, n_samples).tolist(),
        'Temperature': (np.random.normal(98.6, 1, n_samples)).tolist(),
        'OxygenSaturation': (np.random.normal(97, 2, n_samples)).tolist(),
        'Weight': (np.random.normal(70, 15, n_samples)).tolist(),
        'Height': (np.random.normal(170, 10, n_samples)).tolist(),
        'BMI': (np.random.normal(25, 5, n_samples)).tolist(),
        'FamilyHistory': np.random.randint(0, 2, n_samples).tolist(),
        'Outcome': np.random.randint(0, 2, n_samples).tolist()
    }
    
    return data

def save_all_disease_datasets():
    """Save all disease datasets to the database"""
    # Initialize additional disease datasets
    initialize_additional_disease_datasets()
    
    for disease_name, disease_info in DISEASE_DATASETS.items():
        # Create DataFrame from data
        df = pd.DataFrame(disease_info['data'])
        
        # Save to database
        database.save_disease_dataset(disease_name, df)
        
        # Create and save model metadata
        features = disease_info['features']
        metadata = {
            'name': disease_name,
            'features': features,
            'target': disease_info['target'],
            'description': disease_info['description'],
            'metrics': {
                'accuracy': 0.85,  # These will be updated when we train models
                'precision': 0.83,
                'recall': 0.81,
                'f1_score': 0.82
            }
        }
        database.save_model_metadata(disease_name, metadata)

def train_disease_model(disease_name):
    """Train machine learning models for a specific disease
    
    Args:
        disease_name (str): The name of the disease to train models for
        
    Returns:
        dict: Trained models and scaler for the disease
    """
    # Load disease dataset from database
    df = database.load_disease_dataset(disease_name)
    
    if df is None:
        # Dataset doesn't exist, use the default one if available
        if disease_name in DISEASE_DATASETS:
            df = pd.DataFrame(DISEASE_DATASETS[disease_name]['data'])
            # Save it to the database for future use
            database.save_disease_dataset(disease_name, df)
        else:
            # Disease not supported
            return None
    
    # Load metadata to get features and target information
    metadata = database.load_model_metadata(disease_name)
    
    if metadata is None:
        # Metadata doesn't exist, use default if available
        if disease_name in DISEASE_DATASETS:
            metadata = {
                'name': disease_name,
                'features': DISEASE_DATASETS[disease_name]['features'],
                'target': DISEASE_DATASETS[disease_name]['target'],
                'description': DISEASE_DATASETS[disease_name]['description']
            }
            # Save it to the database for future use
            database.save_model_metadata(disease_name, metadata)
        else:
            # Disease not supported
            return None
    
    # Extract features and target
    features = metadata['features']
    target = metadata['target']
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train SVM model
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Initialize and train Logistic Regression model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # Initialize and train Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    models = {'svm': svm_model, 'lr': lr_model, 'rf': rf_model}
    metrics = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
    
    # Update metadata with model metrics
    metadata['metrics'] = {
        'accuracy': np.mean([m['accuracy'] for m in metrics.values()]),
        'precision': np.mean([m['precision'] for m in metrics.values()]),
        'recall': np.mean([m['recall'] for m in metrics.values()]),
        'f1_score': np.mean([m['f1_score'] for m in metrics.values()])
    }
    database.save_model_metadata(disease_name, metadata)
    
    # Return models, scaler, and feature list
    return {
        'svm': svm_model,
        'lr': lr_model,
        'rf': rf_model,
        'scaler': scaler,
        'features': features
    }

def init_models():
    """Initialize and train the machine learning models for all diseases"""
    # Initialize disease models container if it doesn't exist
    if 'disease_models' not in st.session_state:
        st.session_state.disease_models = {}
    
    # Initialize current disease if it doesn't exist
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = 'Diabetes'  # Default disease
    
    # Save all disease datasets to the database if they don't exist
    save_all_disease_datasets()
    
    # Get all available diseases from database
    diseases = database.list_available_diseases()
    
    # If the database is empty, use default diseases
    if not diseases:
        diseases = list(DISEASE_DATASETS.keys()) + ADDITIONAL_DISEASES
    
    # Initialize models for default disease
    if st.session_state.current_disease not in st.session_state.disease_models:
        st.session_state.disease_models[st.session_state.current_disease] = train_disease_model(st.session_state.current_disease)

def get_available_diseases():
    """Get list of available diseases
    
    Returns:
        list: List of available disease names
    """
    # Get all available diseases from database
    diseases = database.list_available_diseases()
    
    # If the database is empty, use default diseases
    if not diseases:
        diseases = list(DISEASE_DATASETS.keys()) + ADDITIONAL_DISEASES
    
    return diseases

def set_current_disease(disease_name):
    """Set the current disease for prediction
    
    Args:
        disease_name (str): The name of the disease
    """
    if disease_name not in st.session_state.disease_models:
        # Load or train models for this disease
        st.session_state.disease_models[disease_name] = train_disease_model(disease_name)
    
    # Set current disease
    st.session_state.current_disease = disease_name

def get_required_features(disease_name=None):
    """Get the required features for a disease
    
    Args:
        disease_name (str, optional): The name of the disease. If None, use current disease.
        
    Returns:
        list: List of required feature names
    """
    if disease_name is None:
        disease_name = st.session_state.current_disease
    
    # Get feature list from metadata
    metadata = database.load_model_metadata(disease_name)
    
    if metadata and 'features' in metadata:
        return metadata['features']
    
    # Fallback to models if available
    if disease_name in st.session_state.disease_models:
        return st.session_state.disease_models[disease_name]['features']
    
    # Use default features from dataset if available
    if disease_name in DISEASE_DATASETS:
        return DISEASE_DATASETS[disease_name]['features']
    
    # Return common health metrics as fallback
    return [
        'Age', 'Sex', 'BloodPressure', 'HeartRate', 
        'RespiratoryRate', 'Temperature', 'OxygenSaturation',
        'Weight', 'Height', 'BMI', 'FamilyHistory'
    ]

def run_prediction(input_data, disease_name=None):
    """Run prediction using the trained models for a specific disease
    
    Args:
        input_data (dict): Dictionary containing the input features
        disease_name (str, optional): The name of the disease. If None, use current disease.
        
    Returns:
        dict: Prediction results from all models
    """
    # Use current disease if not specified
    if disease_name is None:
        disease_name = st.session_state.current_disease
    
    # Check if disease models are initialized
    if 'disease_models' not in st.session_state:
        init_models()
    
    # Check if disease models exist
    if disease_name not in st.session_state.disease_models:
        set_current_disease(disease_name)
    
    # Get models for the disease
    models = st.session_state.disease_models[disease_name]
    
    if models is None:
        return {
            'error': f"No models available for {disease_name}",
            'svm_prediction': 0,
            'svm_probability': 0.0,
            'lr_prediction': 0,
            'lr_probability': 0.0,
            'rf_prediction': 0,
            'rf_probability': 0.0
        }
    
    # Get required features
    required_features = models['features']
    
    # Check if input data has all required features
    for feature in required_features:
        if feature not in input_data:
            # Use a default value if feature is missing
            input_data[feature] = 0
    
    # Filter input data to include only required features
    filtered_input = {feature: input_data[feature] for feature in required_features}
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([filtered_input])
    
    # Standardize input data
    scaler = models['scaler']
    input_scaled = scaler.transform(input_df)
    
    # Get predictions from all models
    svm_model = models['svm']
    lr_model = models['lr']
    rf_model = models['rf']
    
    # SVM prediction
    svm_pred = svm_model.predict(input_scaled)[0]
    svm_prob = svm_model.predict_proba(input_scaled)[0][1]  # Probability of positive class
    
    # Logistic Regression prediction
    lr_pred = lr_model.predict(input_scaled)[0]
    lr_prob = lr_model.predict_proba(input_scaled)[0][1]  # Probability of positive class
    
    # Random Forest prediction
    rf_pred = rf_model.predict(input_scaled)[0]
    rf_prob = rf_model.predict_proba(input_scaled)[0][1]  # Probability of positive class
    
    # Return results
    return {
        'disease': disease_name,
        'svm_prediction': int(svm_pred),
        'svm_probability': float(svm_prob),
        'lr_prediction': int(lr_pred),
        'lr_probability': float(lr_prob),
        'rf_prediction': int(rf_pred),
        'rf_probability': float(rf_prob)
    }
