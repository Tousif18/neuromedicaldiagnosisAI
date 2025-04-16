import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import base64
from datetime import datetime
import file_handlers

# CSS for profile page
def profile_css():
    return """
    <style>
    .profile-container {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.8), rgba(22, 27, 34, 0.8));
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #FF00FF;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
        margin-bottom: 25px;
    }
    
    .add-form-container {
        background: rgba(0, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #00FFFF;
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
    
    .profile-card {
        background: rgba(13, 17, 23, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #00FFFF;
        transition: all 0.3s ease;
    }
    
    .profile-card:hover {
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
    
    .profile-selector {
        background: rgba(255, 0, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #FF00FF;
    }
    
    .file-item {
        background: rgba(13, 17, 23, 0.8);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #00FFFF;
        transition: all 0.3s ease;
    }
    
    .file-item:hover {
        transform: translateX(5px);
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    .file-icon {
        font-size: 1.5em;
        margin-right: 10px;
        color: #FF00FF;
    }
    
    .file-name {
        color: #00FFFF;
        font-weight: bold;
    }
    
    .file-info {
        color: rgba(0, 255, 255, 0.7);
        font-size: 0.9em;
        margin-top: 5px;
        font-family: monospace;
    }
    
    .data-table {
        margin-top: 15px;
        background: rgba(13, 17, 23, 0.5);
        border-radius: 8px;
        border: 1px solid #00FFFF;
        overflow: hidden;
    }
    
    .add-button {
        background: linear-gradient(90deg, #FF00FF, #00FFFF);
        color: #0D1117;
        font-weight: bold;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    
    .add-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.5);
    }
    
    .delete-button {
        background: linear-gradient(90deg, #FF0000, #FF00FF);
        color: white;
        border: none;
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .delete-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 0, 0, 0.5);
    }
    
    .upload-area {
        background: rgba(255, 0, 255, 0.05);
        border: 2px dashed #FF00FF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 15px 0;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        background: rgba(255, 0, 255, 0.1);
    }
    
    .param-label {
        color: #FF00FF;
        font-weight: bold;
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
    </style>
    """

def profile_page():
    """Render the profile page for managing individuals and their data with cyberpunk styling"""
    # Apply custom CSS
    st.markdown(profile_css(), unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("Please login to access this page")
        return
    
    # Title with cyberpunk styling
    st.markdown('<h1 class="cyberpunk-title">Neural Profile Vault</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cyberpunk-subtitle">Manage Medical Identities & Records</p>', unsafe_allow_html=True)
    
    # Add futuristic banner image
    st.image("assets/images/profile_header.png", use_column_width=True)
    
    user_data = st.session_state.users[st.session_state.current_user]
    
    # Create a container for the profile management
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    
    # Current time display with cyberpunk style
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<p style="text-align:right; color:#00FFFF; font-family:monospace; font-size:0.8em;">SYSTEM TIME: {current_time}</p>', unsafe_allow_html=True)
    
    # Section for adding a new individual with cyberpunk styling
    st.markdown('<div class="add-form-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#00FFFF; border-bottom:1px solid #00FFFF; padding-bottom:5px;">🧬 Register New Identity</h3>', unsafe_allow_html=True)
    
    with st.form("add_individual_form"):
        st.markdown('<p class="param-label">Name or Alias</p>', unsafe_allow_html=True)
        name = st.text_input("", placeholder="Enter a unique name for this individual")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<p style="color:#00FFFF; font-size:0.9em; margin-top:5px;">This identifier will be used throughout the system</p>', unsafe_allow_html=True)
        with col2:
            submitted = st.form_submit_button("CREATE PROFILE")
        
        if submitted:
            if name:
                if name in user_data["individuals"]:
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <span style="color:#FF0000; font-weight:bold;">❌ ERROR:</span> Individual '{name}' already exists in the system.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    user_data["individuals"][name] = {
                        "name": name,
                        "data": {},
                        "files": {}
                    }
                    user_data["diagnosis_history"][name] = []
                    st.markdown(
                        f"""
                        <div class="success-box">
                            <span style="color:#00FF00; font-weight:bold;">✅ SUCCESS:</span> Individual '{name}' successfully registered in the system.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.rerun()
            else:
                st.markdown(
                    """
                    <div class="warning-box">
                        <span style="color:#FF0000; font-weight:bold;">⚠️ WARNING:</span> Please enter a name or alias for the individual.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close add form container
    
    # Section for managing individuals with cyberpunk styling
    st.markdown('<h3 style="color:#FF00FF; border-bottom:1px solid #FF00FF; padding-bottom:5px; margin-top:30px;">👥 Manage Neural Profiles</h3>', unsafe_allow_html=True)
    
    if not user_data["individuals"]:
        st.markdown(
            """
            <div class="info-box">
                <span style="color:#00FFFF; font-weight:bold;">ℹ️ INFORMATION:</span> No individuals registered yet. Use the form above to add someone to the system.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Individual selector with cyberpunk styling
        st.markdown('<div class="profile-selector">', unsafe_allow_html=True)
        st.markdown('<p style="color:#FF00FF; font-size:1.1em; margin-bottom:10px;">👤 Select Profile to Manage</p>', unsafe_allow_html=True)
        
        # Get list of individuals with count of records
        individuals = list(user_data["individuals"].keys())
        
        # Create formatted options with record counts
        def format_individual(name):
            individual = user_data["individuals"][name]
            file_count = len(individual["files"])
            data_entries = len(individual["data"].get("manual_entries", []))
            return f"{name} [{file_count} files, {data_entries} data entries]"
        
        selected_individual = st.selectbox("", individuals, format_func=format_individual)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_individual:
            individual_data = user_data["individuals"][selected_individual]
            
            # Profile header with stats
            file_count = len(individual_data["files"])
            data_entries = len(individual_data["data"].get("manual_entries", []))
            
            st.markdown(
                f"""
                <div style="background:rgba(255,0,255,0.1); padding:15px; border-radius:5px; margin:15px 0; text-align:center; border:1px solid #FF00FF;">
                    <span style="font-size:1.5em; color:#00FFFF;">Profile: <span style="color:#FF00FF; font-weight:bold;">{selected_individual}</span></span>
                    <div style="margin-top:10px; display:flex; justify-content:center; gap:20px;">
                        <span style="color:#00FFFF; padding:5px 10px; background:rgba(255,0,255,0.2); border-radius:5px;">
                            <i class="fas fa-file"></i> {file_count} Files
                        </span>
                        <span style="color:#00FFFF; padding:5px 10px; background:rgba(255,0,255,0.2); border-radius:5px;">
                            <i class="fas fa-database"></i> {data_entries} Data Entries
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Two tabs for different sections
            tab1, tab2, tab3 = st.tabs(["📤 Upload Medical Data", "✏️ Manual Data Entry", "📋 View Records"])
            
            # Tab 1: File Upload
            with tab1:
                st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">📤 Upload Medical Records</h3>', unsafe_allow_html=True)
                
                # Cyberpunk-styled upload area
                st.markdown(
                    """
                    <div class="upload-area">
                        <div style="font-size:2em; color:#FF00FF; margin-bottom:10px;">📁</div>
                        <div style="color:#00FFFF; margin-bottom:10px;">Drag and drop medical files here</div>
                        <div style="color:#FF00FF; font-size:0.8em;">Supported formats: PDF, CSV, TXT, DOCX, JSON</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                uploaded_file = st.file_uploader(
                    "Upload medical records", 
                    type=["pdf", "csv", "txt", "docx", "json"],
                    label_visibility="collapsed"
                )
                
                if uploaded_file:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    file_content = uploaded_file.read()
                    
                    # File type validation with appropriate icons
                    file_icons = {
                        "pdf": "📕",
                        "csv": "📊",
                        "txt": "📝",
                        "docx": "📄",
                        "json": "🔧"
                    }
                    
                    file_icon = file_icons.get(file_ext, "📁")
                    
                    # Display file info
                    st.markdown(
                        f"""
                        <div class="file-item">
                            <div><span class="file-icon">{file_icon}</span> <span class="file-name">{uploaded_file.name}</span></div>
                            <div class="file-info">Type: {file_ext.upper()} | Size: {len(file_content)/1024:.1f} KB</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Processing animation
                    with st.spinner("Processing file..."):
                        # Save the file content
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        file_id = f"{timestamp}_{uploaded_file.name}"
                        
                        individual_data["files"][file_id] = {
                            "filename": uploaded_file.name,
                            "content_type": file_ext,
                            "upload_date": timestamp,
                        }
                        
                        # Process file content based on type
                        try:
                            extracted_data = file_handlers.process_file(file_content, file_ext)
                            
                            if extracted_data:
                                st.success("File uploaded and processed successfully!")
                                
                                # Store extracted data if available
                                if isinstance(extracted_data, dict):
                                    for key, value in extracted_data.items():
                                        if key not in individual_data["data"]:
                                            individual_data["data"][key] = []
                                        individual_data["data"][key].append(value)
                                
                                # Display extracted data preview
                                st.markdown('<h4 style="color:#00FFFF; margin-top:15px;">Extracted Data Preview:</h4>', unsafe_allow_html=True)
                                st.json(extracted_data)
                            else:
                                st.info("File uploaded, but no structured data could be extracted. The file is saved for reference.")
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close profile card
            
            # Tab 2: Manual Data Entry
            with tab2:
                st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">✏️ Manual Data Entry</h3>', unsafe_allow_html=True)
                
                with st.form(f"manual_data_{selected_individual}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<p class="param-label">Glucose Level (mg/dL)</p>', unsafe_allow_html=True)
                        glucose = st.number_input("", 
                                                min_value=0.0, 
                                                help="Normal range: 70-140 mg/dL")
                        
                        st.markdown('<p class="param-label">Blood Pressure (mmHg)</p>', unsafe_allow_html=True)
                        blood_pressure = st.number_input("", 
                                                        min_value=0.0,
                                                        key="bp",
                                                        help="Normal range: 90-120 mmHg")
                        
                        st.markdown('<p class="param-label">Skin Thickness (mm)</p>', unsafe_allow_html=True)
                        skin_thickness = st.number_input("", 
                                                        min_value=0.0,
                                                        key="skin",
                                                        help="Normal range: 20-40 mm")
                        
                        st.markdown('<p class="param-label">Insulin Level (mU/L)</p>', unsafe_allow_html=True)
                        insulin = st.number_input("", 
                                                min_value=0.0,
                                                key="insulin",
                                                help="Normal range: 16-166 mU/L")
                    
                    with col2:
                        st.markdown('<p class="param-label">BMI</p>', unsafe_allow_html=True)
                        bmi = st.number_input("", 
                                            min_value=0.0,
                                            key="bmi",
                                            help="Normal range: 18.5-24.9")
                        
                        st.markdown('<p class="param-label">Diabetes Pedigree Function</p>', unsafe_allow_html=True)
                        diabetes_pedigree = st.number_input("", 
                                                            min_value=0.0,
                                                            key="dpf",
                                                            help="Likelihood of diabetes based on family history")
                        
                        st.markdown('<p class="param-label">Age</p>', unsafe_allow_html=True)
                        age = st.number_input("", 
                                            min_value=0, 
                                            max_value=120,
                                            key="age")
                        
                        st.markdown(
                            """
                            <div style="background:rgba(0,255,255,0.05); padding:10px; border-radius:5px; margin-top:15px; border:1px solid #00FFFF;">
                                <p style="color:#00FFFF; font-size:0.9em;">These values will be used for AI diagnosis. Complete all fields for best results.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Submit button with custom styling
                    submitted = st.form_submit_button("SAVE MEDICAL DATA")
                    
                    if submitted:
                        # Validation
                        if glucose == 0.0 and blood_pressure == 0.0 and bmi == 0.0 and age == 0:
                            st.markdown(
                                """
                                <div class="warning-box">
                                    <span style="color:#FF0000; font-weight:bold;">⚠️ WARNING:</span> Please enter at least some values to save.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            # Save manual data to individual's profile
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            manual_data = {
                                "timestamp": timestamp,
                                "glucose": glucose,
                                "blood_pressure": blood_pressure,
                                "skin_thickness": skin_thickness,
                                "insulin": insulin,
                                "bmi": bmi,
                                "diabetes_pedigree": diabetes_pedigree,
                                "age": age
                            }
                            
                            # Add to individual's data
                            if "manual_entries" not in individual_data["data"]:
                                individual_data["data"]["manual_entries"] = []
                            
                            individual_data["data"]["manual_entries"].append(manual_data)
                            
                            st.markdown(
                                """
                                <div class="success-box">
                                    <span style="color:#00FF00; font-weight:bold;">✅ SUCCESS:</span> Medical data saved successfully to profile.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close profile card
            
            # Tab 3: View Medical Records
            with tab3:
                st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">📋 Medical Records</h3>', unsafe_allow_html=True)
                
                # Files section
                if individual_data["files"]:
                    st.markdown('<h4 style="color:#00FFFF; margin-top:15px;">Uploaded Files</h4>', unsafe_allow_html=True)
                    
                    # Get file extensions for icons
                    file_icons = {
                        "pdf": "📕",
                        "csv": "📊",
                        "txt": "📝",
                        "docx": "📄",
                        "json": "🔧"
                    }
                    
                    # Display files with cyberpunk styling
                    for file_id, file_info in individual_data["files"].items():
                        file_ext = file_info['content_type'].lower()
                        file_icon = file_icons.get(file_ext, "📁")
                        
                        upload_date = file_info['upload_date']
                        formatted_date = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]} {upload_date[8:10]}:{upload_date[10:12]}:{upload_date[12:14]}"
                        
                        st.markdown(
                            f"""
                            <div class="file-item">
                                <div><span class="file-icon">{file_icon}</span> <span class="file-name">{file_info['filename']}</span></div>
                                <div class="file-info">Type: {file_info['content_type'].upper()} | Upload Date: {formatted_date}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        """
                        <div class="info-box">
                            <span style="color:#00FFFF; font-weight:bold;">ℹ️ INFORMATION:</span> No files uploaded for this individual yet.
                            <p style="margin-top:5px;">Upload files in the "Upload Medical Data" tab.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Manual data entries section
                if "manual_entries" in individual_data["data"] and individual_data["data"]["manual_entries"]:
                    st.markdown('<h4 style="color:#00FFFF; margin-top:25px;">Manual Data Entries</h4>', unsafe_allow_html=True)
                    
                    # Convert list of dictionaries to DataFrame for better display
                    df = pd.DataFrame(individual_data["data"]["manual_entries"])
                    
                    # Sort by timestamp (newest first)
                    if "timestamp" in df.columns:
                        df = df.sort_values(by="timestamp", ascending=False)
                    
                    st.dataframe(df, use_column_width=True)
                    
                    # Add download option
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    filename = f"{selected_individual}_medical_data.csv"
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;"><div style="background:linear-gradient(90deg, #00FFFF, #0088FF); color:#0D1117; text-align:center; padding:8px; border-radius:5px; margin-top:10px; font-weight:bold; cursor:pointer;">📥 DOWNLOAD DATA AS CSV</div></a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.markdown(
                        """
                        <div class="info-box">
                            <span style="color:#00FFFF; font-weight:bold;">ℹ️ INFORMATION:</span> No manual data entries for this individual yet.
                            <p style="margin-top:5px;">Add data entries in the "Manual Data Entry" tab.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close profile card
            
            # Add delete option at the bottom
            st.markdown('<div style="margin-top:30px; text-align:center;">', unsafe_allow_html=True)
            
            if st.button(f"🗑️ DELETE {selected_individual.upper()} PROFILE", help="Permanently delete this profile and all associated data"):
                confirmation = st.checkbox("I understand this action cannot be undone and will delete all data for this profile")
                if confirmation:
                    del user_data["individuals"][selected_individual]
                    if selected_individual in user_data["diagnosis_history"]:
                        del user_data["diagnosis_history"][selected_individual]
                    
                    st.markdown(
                        f"""
                        <div class="success-box" style="text-align:center">
                            <span style="color:#00FF00; font-weight:bold;">✅ SUCCESS:</span> Profile for {selected_individual} has been permanently deleted.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the profile container
    st.markdown('</div>', unsafe_allow_html=True)
