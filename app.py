import streamlit as st
import pandas as pd
import numpy as np
import os
import auth
import profile
import diagnosis
import ml_models
import utils

# Set page configuration
st.set_page_config(
    page_title="AI Medical Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'init_app' not in st.session_state:
    # Initialize the ML models
    ml_models.init_models()
    st.session_state.init_app = True

# Main app sidebar navigation
def sidebar_navigation():
    # Custom CSS for sidebar
    st.markdown(
        """
        <style>
        .sidebar-title {
            background: linear-gradient(to right, #FF00FF, #00FFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-align: center;
        }
        .nav-button {
            background: linear-gradient(90deg, rgba(255,0,255,0.1), rgba(0,255,255,0.1));
            border: 1px solid;
            border-image: linear-gradient(to right, #FF00FF, #00FFFF);
            border-image-slice: 1;
            color: #00FFFF !important;
            text-align: center;
            margin: 5px 0;
            transition: all 0.3s;
            border-radius: 5px;
            font-weight: bold;
        }
        .nav-button:hover {
            background: linear-gradient(90deg, rgba(255,0,255,0.3), rgba(0,255,255,0.3));
            box-shadow: 0 0 10px #FF00FF;
            transform: translateY(-2px);
        }
        .user-welcome {
            background: rgba(0,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            text-align: center;
            border-left: 3px solid #00FFFF;
        }
        .sidebar-footer {
            position: fixed;
            bottom: 0;
            text-align: center;
            width: 100%;
            background: linear-gradient(90deg, #0D1117, #161B22);
            padding: 10px 0;
            font-size: 0.8em;
            color: #00FFFF;
            border-top: 1px solid #FF00FF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Add logo/branding at the top of sidebar
    st.sidebar.markdown('<div class="sidebar-title">NeurodiagnostAI</div>', unsafe_allow_html=True)
    st.sidebar.image("assets/images/neural_scan.png", use_column_width=True)
    st.sidebar.markdown("---")
    
    if st.session_state.logged_in:
        # User welcome message with cyberpunk styling
        st.sidebar.markdown(f'<div class="user-welcome">👤 Welcome, <span style="color:#FF00FF; font-weight:bold">{st.session_state.current_user}</span>!</div>', unsafe_allow_html=True)
        
        # Custom-styled navigation buttons
        if st.sidebar.button("🏠 Home", key="home_btn", help="Return to the main dashboard"):
            st.session_state.page = 'home'
        
        if st.sidebar.button("👥 Profile Management", key="profile_btn", help="Manage individuals and their medical data"):
            st.session_state.page = 'profile'
        
        if st.sidebar.button("🔬 New Diagnosis", key="diagnosis_btn", help="Run a new medical diagnosis"):
            st.session_state.page = 'diagnosis'
        
        if st.sidebar.button("📋 Diagnosis History", key="history_btn", help="View past diagnosis results"):
            st.session_state.page = 'history'
        
        # Logout button with different styling
        if st.sidebar.button("🚪 Logout", key="logout_btn", help="Sign out of your account"):
            auth.logout()
    else:
        # Login info message with custom styling
        st.sidebar.markdown("""
            <div style="background:rgba(255,0,255,0.1); padding:10px; border-radius:5px; text-align:center; margin-bottom:15px; border:1px solid #FF00FF;">
                <span style="color:#00FFFF">Please login to access the application</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Login and register buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔑 Login", key="login_btn", help="Sign in to your account"):
                st.session_state.page = 'login'
        with col2:
            if st.button("✨ Register", key="register_btn", help="Create a new account"):
                st.session_state.page = 'register'
    
    # Footer with version and copyright
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="text-align:center;">
            <div style="margin-bottom:5px;">
                <span style="color:#FF00FF; font-size:0.9em;">v2.0.0</span>
            </div>
            <div>
                <span style="color:#00FFFF; font-size:0.8em;">© 2025 NeurodiagnostAI</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Home page
def home_page():
    # Create a more visually striking title with custom HTML and CSS
    st.markdown(
        """
        <style>
        .title-container {
            background: linear-gradient(90deg, #0e1c26, #2a3c54);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            border: 2px solid #00FFFF;
            box-shadow: 0 0 15px #FF00FF;
        }
        .cyberpunk-title {
            font-family: 'Arial Black', sans-serif;
            font-size: 3.2em;
            font-weight: 900;
            background: linear-gradient(to right, #FF00FF, #00FFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .cyberpunk-subtitle {
            color: #00FFFF;
            font-size: 1.5em;
            margin-top: 0;
            letter-spacing: 1px;
        }
        .feature-container {
            background: rgba(22, 27, 34, 0.8);
            border-left: 4px solid #FF00FF;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .feature-container:hover {
            transform: translateX(5px);
            box-shadow: 0 0 10px #FF00FF;
        }
        .feature-title {
            color: #FF00FF;
            font-weight: bold;
            font-size: 1.2em;
        }
        .get-started-container {
            background: linear-gradient(45deg, #161B22, #1A1F2C);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #00FFFF;
            margin-top: 25px;
        }
        .step-number {
            background: #FF00FF;
            color: #0D1117;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 50%;
            margin-right: 10px;
            display: inline-block;
        }
        .disclaimer {
            background: rgba(255, 0, 255, 0.1);
            border-left: 4px solid #FF00FF;
            padding: 15px;
            margin-top: 30px;
            border-radius: 5px;
        }
        </style>
        <div class="title-container">
            <h1 class="cyberpunk-title">NEURODIAGNOST.AI</h1>
            <p class="cyberpunk-subtitle">Advanced Medical Diagnostics Powered by Artificial Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Banner image (cyberpunk-style medical visualization)
    st.image("assets/images/brain_scan.png", use_column_width=True)
    
    # Introduction text
    st.markdown("""
    ## 🧠 Revolutionizing Medical Diagnostics
    
    **NeurodiagnostAI** leverages cutting-edge machine learning models to analyze medical parameters 
    and predict potential health conditions with unprecedented accuracy. Our system follows the complete 
    data processing pipeline from medical data intake to refined predictions.
    """)
    
    # Features section with improved styling
    st.markdown('<h3 style="color:#FF00FF; border-bottom:2px solid #00FFFF; padding-bottom:8px;">🔮 KEY FEATURES</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-container"><span class="feature-title">🧬 Multi-Profile Management</span><br>Upload and manage medical data for multiple individuals under a single account</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-container"><span class="feature-title">📊 Advanced AI Models</span><br>Utilizes SVM, Logistic Regression, and Random Forest algorithms for comprehensive diagnosis</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-container"><span class="feature-title">📁 Multi-Format Support</span><br>Upload medical records in PDF, CSV, TXT, DOCX, or JSON format with automatic data extraction</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-container"><span class="feature-title">🔒 Secure Data Storage</span><br>All your medical data is securely stored and private</div>', unsafe_allow_html=True)
    
    # Get Started section with steps
    st.markdown('<div class="get-started-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#00FFFF; text-align:center; margin-bottom:20px;">🚀 GET STARTED</h3>', unsafe_allow_html=True)
    
    st.markdown('<p><span class="step-number">1</span> Go to the <b>Profile</b> section to add individuals</p>', unsafe_allow_html=True)
    st.markdown('<p><span class="step-number">2</span> Upload their medical records or enter data manually</p>', unsafe_allow_html=True)
    st.markdown('<p><span class="step-number">3</span> Navigate to <b>New Diagnosis</b> to run predictions</p>', unsafe_allow_html=True)
    st.markdown('<p><span class="step-number">4</span> View and export past diagnoses in <b>Diagnosis History</b></p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown('<div class="disclaimer">⚠️ <b>DISCLAIMER:</b> This system is not a replacement for professional medical advice. Always consult with healthcare professionals for proper diagnosis and treatment.</div>', unsafe_allow_html=True)

# Main application logic
def main():
    sidebar_navigation()
    
    # Route to the appropriate page based on session state
    if not st.session_state.logged_in:
        if st.session_state.page == 'login':
            auth.login_page()
        elif st.session_state.page == 'register':
            auth.register_page()
    else:
        if st.session_state.page == 'home':
            home_page()
        elif st.session_state.page == 'profile':
            profile.profile_page()
        elif st.session_state.page == 'diagnosis':
            diagnosis.diagnosis_page()
        elif st.session_state.page == 'history':
            diagnosis.history_page()

if __name__ == "__main__":
    main()
