# NeurodiagnostAI - AI-Based Medical Diagnosis Web Application

## Project Overview
NeurodiagnostAI is a comprehensive medical diagnosis web application powered by artificial intelligence. The application enables users to manage multiple individuals' medical records and receive ML-powered disease predictions with a futuristic cyberpunk user interface.

## Key Features
- **User Authentication**: Secure login and registration system with data persistence
- **Multi-Profile Management**: Manage medical data for multiple individuals under a single account
- **AI-Powered Diagnosis**: Advanced machine learning models (SVM, Logistic Regression, Random Forest) for disease prediction
- **Multiple Disease Support**: Analysis for up to 20 different medical conditions
- **Medical Record Management**: Upload and process medical data from various file formats (PDF, CSV, TXT, DOCX, JSON)
- **Diagnostic History**: Track and analyze diagnosis results over time with visualizations
- **Data Visualization**: Interactive and visually appealing representations of medical data
- **Secure Local Storage**: All data stored securely on the local device
- **Futuristic Cyberpunk UI**: Modern, engaging user interface with neon aesthetics

## Technical Implementation
- **Frontend/Backend**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (SVM, Logistic Regression, Random Forest algorithms)
- **File Processing**: PyMuPDF, PyPDF2 for document handling
- **Data Storage**: Local file-based storage system (JSON)
- **UI/UX**: Custom CSS styling with cyberpunk theme

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (streamlit, numpy, pandas, scikit-learn, pymupdf, pypdf2)

### Installation
1. Clone the repository
```
git clone https://github.com/your-username/neurodiagnostai.git
cd neurodiagnostai
```

2. Install required packages
```
pip install streamlit numpy pandas scikit-learn pymupdf pypdf2
```

3. Run the application
```
streamlit run app.py
```

4. Access the application in your web browser
```
http://localhost:50005000
```

## User Workflow
1. **Register/Login**: Create a new account or log in to an existing account
2. **Profile Management**: Add individuals and their medical data in the "Profile Management" section
3. **Upload Medical Records**: Upload medical files or enter data manually for each individual
4. **Run Diagnosis**: Select an individual and disease type in the "New Diagnosis" section
5. **View Results**: Get immediate diagnostic results with confidence levels and model details
6. **Track History**: Access past diagnostic results in the "Diagnosis History" section
7. **Export Data**: Export diagnosis history to CSV for external analysis

## Project Structure
- **app.py**: Main application entry point and UI routing
- **auth.py**: Authentication functionality (login, registration, security)
- **profile.py**: Profile management for individuals and their data
- **diagnosis.py**: Diagnostic functionality and results visualization
- **ml_models.py**: Machine learning model implementation and training
- **database.py**: Data storage and retrieval functionality
- **file_handlers.py**: Processing and data extraction from various file formats
- **utils.py**: Utility functions and helpers

## Security Features
- Password hashing for secure authentication
- Session-based user management
- Data validation for all inputs
- Secure local storage of sensitive medical information

## Disclaimer
This application is for educational and demonstration purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for any medical concerns.