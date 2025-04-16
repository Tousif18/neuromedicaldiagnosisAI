import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Define the directory for database storage
DB_DIR = "database"
USERS_FILE = os.path.join(DB_DIR, "users.json")
DISEASE_MODELS_DIR = os.path.join(DB_DIR, "models")
DISEASE_DATA_DIR = os.path.join(DB_DIR, "datasets")

# Initialize database
def init_database():
    """Initialize the database directory structure"""
    # Create main database directory
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    
    # Create directory for disease models
    if not os.path.exists(DISEASE_MODELS_DIR):
        os.makedirs(DISEASE_MODELS_DIR)
    
    # Create directory for disease datasets
    if not os.path.exists(DISEASE_DATA_DIR):
        os.makedirs(DISEASE_DATA_DIR)
    
    # Initialize users file if it doesn't exist
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)

# User data operations
def save_users(users_data):
    """Save users data to the database file
    
    Args:
        users_data (dict): The users data to save
    """
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f, indent=2)

def load_users():
    """Load users data from the database file
    
    Returns:
        dict: The users data
    """
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Return empty dict if file is corrupt
                return {}
    return {}

# Disease data operations
def save_disease_dataset(disease_name, dataset):
    """Save a disease dataset to the database
    
    Args:
        disease_name (str): The name of the disease
        dataset (pd.DataFrame): The dataset to save
    """
    # Create safe filename from disease name
    safe_name = "".join(c if c.isalnum() else "_" for c in disease_name.lower())
    file_path = os.path.join(DISEASE_DATA_DIR, f"{safe_name}_data.csv")
    
    # Save dataset to CSV
    dataset.to_csv(file_path, index=False)

def load_disease_dataset(disease_name):
    """Load a disease dataset from the database
    
    Args:
        disease_name (str): The name of the disease
        
    Returns:
        pd.DataFrame or None: The loaded dataset or None if not found
    """
    # Create safe filename from disease name
    safe_name = "".join(c if c.isalnum() else "_" for c in disease_name.lower())
    file_path = os.path.join(DISEASE_DATA_DIR, f"{safe_name}_data.csv")
    
    # Load dataset from CSV if it exists
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def list_available_diseases():
    """List all available diseases in the database
    
    Returns:
        list: List of available disease names
    """
    disease_files = []
    if os.path.exists(DISEASE_DATA_DIR):
        for file in os.listdir(DISEASE_DATA_DIR):
            if file.endswith("_data.csv"):
                disease_name = file.replace("_data.csv", "")
                # Convert back to readable format (replace underscores with spaces and capitalize)
                readable_name = " ".join(word.capitalize() for word in disease_name.split("_"))
                disease_files.append(readable_name)
    return disease_files

# Model operations
def save_model_metadata(disease_name, model_metadata):
    """Save model metadata for a disease
    
    Args:
        disease_name (str): The name of the disease
        model_metadata (dict): The model metadata (features, accuracy, etc.)
    """
    # Create safe filename from disease name
    safe_name = "".join(c if c.isalnum() else "_" for c in disease_name.lower())
    file_path = os.path.join(DISEASE_MODELS_DIR, f"{safe_name}_metadata.json")
    
    # Save metadata to JSON
    with open(file_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)

def load_model_metadata(disease_name):
    """Load model metadata for a disease
    
    Args:
        disease_name (str): The name of the disease
        
    Returns:
        dict or None: The model metadata or None if not found
    """
    # Create safe filename from disease name
    safe_name = "".join(c if c.isalnum() else "_" for c in disease_name.lower())
    file_path = os.path.join(DISEASE_MODELS_DIR, f"{safe_name}_metadata.json")
    
    # Load metadata from JSON if it exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

# Initialize the database when this module is imported
init_database()