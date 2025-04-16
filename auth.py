import streamlit as st
import hashlib
import time
from datetime import datetime

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

# Common CSS for auth pages
def auth_css():
    return """
    <style>
    .auth-container {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.7), rgba(22, 27, 34, 0.7));
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #FF00FF;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
        margin-bottom: 20px;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .auth-title {
        font-family: 'Arial Black', sans-serif;
        font-size: 2.5em;
        font-weight: 900;
        background: linear-gradient(to right, #FF00FF, #00FFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-transform: uppercase;
        text-align: center;
        letter-spacing: 2px;
    }
    
    .auth-subtitle {
        color: #00FFFF;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.2em;
    }
    
    .auth-info {
        background: rgba(0, 255, 255, 0.1);
        border-left: 3px solid #00FFFF;
        padding: 10px 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    
    .auth-button {
        background: linear-gradient(90deg, #FF00FF, #00FFFF);
        color: #0D1117;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        text-align: center;
        margin: 15px 0;
        transition: all 0.3s;
        border: none;
        cursor: pointer;
    }
    
    .auth-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.5);
    }
    
    .form-field {
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid #00FFFF;
        border-radius: 5px;
        color: #00FFFF;
        padding: 10px;
        margin-bottom: 15px;
    }
    
    .auth-footer {
        text-align: center;
        margin-top: 20px;
        color: #00FFFF;
        font-size: 0.8em;
    }
    
    .success-message {
        background: linear-gradient(90deg, rgba(0, 255, 0, 0.1), rgba(0, 255, 255, 0.1));
        border-left: 3px solid #00FF00;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
        text-align: center;
    }
    
    .error-message {
        background: linear-gradient(90deg, rgba(255, 0, 0, 0.1), rgba(255, 0, 255, 0.1));
        border-left: 3px solid #FF0000;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
        text-align: center;
    }
    
    .warning-message {
        background: linear-gradient(90deg, rgba(255, 165, 0, 0.1), rgba(255, 0, 255, 0.1));
        border-left: 3px solid #FFA500;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """

def login_page():
    """Render the login page with cyberpunk styling"""
    # Apply custom CSS
    st.markdown(auth_css(), unsafe_allow_html=True)
    
    # Create container with custom styling
    st.markdown(
        """
        <div class="auth-container">
            <h1 class="auth-title">Secure Login</h1>
            <p class="auth-subtitle">Enter your credentials to access the system</p>
            <div class="auth-info">
                <p>🔐 All connections are encrypted and secure</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add cyber-themed graphics
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/images/login_image.png", use_column_width=True)
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        # Current time display with cyberpunk style
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f'<p style="text-align:right; color:#00FFFF; font-family:monospace; font-size:0.8em;">SYSTEM TIME: {current_time}</p>', unsafe_allow_html=True)
        
        # Submit button with custom color
        submitted = st.form_submit_button("LOGIN")
        
        if submitted:
            if username and password:
                # Check if username exists and password is correct
                if username in st.session_state.users and st.session_state.users[username]["password"] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.session_state.page = 'home'
                    
                    # Success message with custom styling
                    st.markdown(
                        '<div class="success-message">✅ Authentication successful! Redirecting to dashboard...</div>',
                        unsafe_allow_html=True
                    )
                    
                    time.sleep(1)
                    st.rerun()
                else:
                    # Error message with custom styling
                    st.markdown(
                        '<div class="error-message">❌ Authentication failed: Invalid username or password</div>',
                        unsafe_allow_html=True
                    )
            else:
                # Warning message with custom styling
                st.markdown(
                    '<div class="warning-message">⚠️ Please enter both username and password</div>',
                    unsafe_allow_html=True
                )
    
    # Footer with link to register
    st.markdown(
        """
        <div class="auth-footer">
            <p>Don't have an account? Click the Register button in the sidebar</p>
            <p style="margin-top:5px; font-size:0.9em; color:#FF00FF;">NeurodiagnostAI © 2025 | Secure Authentication System v2.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def register_page():
    """Render the registration page with cyberpunk styling"""
    # Apply custom CSS
    st.markdown(auth_css(), unsafe_allow_html=True)
    
    # Create container with custom styling
    st.markdown(
        """
        <div class="auth-container">
            <h1 class="auth-title">Create Account</h1>
            <p class="auth-subtitle">Join the future of medical diagnostics</p>
            <div class="auth-info">
                <p>🔒 Your data is encrypted and protected</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Registration image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/images/register_image.png", use_column_width=True)
    
    # Registration form
    with st.form("register_form"):
        username = st.text_input("Choose Username", placeholder="Create a unique username")
        password = st.text_input("Create Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        # Terms and conditions
        st.markdown(
            """
            <div style="background:rgba(0,255,255,0.05); padding:10px; border-radius:5px; margin:15px 0; border:1px solid #00FFFF;">
                <p style="font-size:0.9em; color:#00FFFF;">By registering, you agree to our Terms of Service and Privacy Policy.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Current time display with cyberpunk style
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f'<p style="text-align:right; color:#00FFFF; font-family:monospace; font-size:0.8em;">SYSTEM TIME: {current_time}</p>', unsafe_allow_html=True)
        
        # Submit button with custom color
        submitted = st.form_submit_button("CREATE ACCOUNT")
        
        if submitted:
            if username and password and confirm_password:
                if username in st.session_state.users:
                    # Error message with custom styling
                    st.markdown(
                        '<div class="error-message">❌ Username already exists. Please choose another one.</div>',
                        unsafe_allow_html=True
                    )
                elif password != confirm_password:
                    # Error message with custom styling
                    st.markdown(
                        '<div class="error-message">❌ Passwords do not match. Please try again.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Create new user
                    st.session_state.users[username] = {
                        "password": hash_password(password),
                        "individuals": {},  # Store individuals' data here
                        "diagnosis_history": {}  # Store diagnosis history here
                    }
                    # Add the user themselves as the first individual
                    st.session_state.users[username]["individuals"]["Self"] = {
                        "name": "Self",
                        "data": {},
                        "files": {}
                    }
                    
                    # Success message with custom styling
                    st.markdown(
                        '<div class="success-message">✅ Registration successful! Redirecting to login...</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.session_state.page = 'login'
                    time.sleep(1)
                    st.rerun()
            else:
                # Warning message with custom styling
                st.markdown(
                    '<div class="warning-message">⚠️ Please fill in all fields</div>',
                    unsafe_allow_html=True
                )
    
    # Footer with link to login
    st.markdown(
        """
        <div class="auth-footer">
            <p>Already have an account? Click the Login button in the sidebar</p>
            <p style="margin-top:5px; font-size:0.9em; color:#FF00FF;">NeurodiagnostAI © 2025 | Secure Registration System v2.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def logout():
    """Log out the current user"""
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.page = 'login'
    st.rerun()
