import streamlit as st
import pandas as pd
import io
import json
import re
import base64

# Import specialized file handlers, handling the case if they're not available
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx2txt
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

def process_file(file_content, file_ext):
    """Process uploaded file based on its extension
    
    Args:
        file_content (bytes): The content of the uploaded file
        file_ext (str): The file extension
        
    Returns:
        dict or None: Extracted data from the file, or None if no data could be extracted
    """
    
    # Process based on file type
    if file_ext == 'csv':
        return process_csv(file_content)
    elif file_ext == 'txt':
        return process_txt(file_content)
    elif file_ext == 'pdf' and PDF_SUPPORT:
        return process_pdf(file_content)
    elif file_ext == 'docx' and DOCX_SUPPORT:
        return process_docx(file_content)
    elif file_ext == 'json':
        return process_json(file_content)
    else:
        st.warning(f"File format {file_ext} is not supported or the required library is not installed.")
        return None

def process_csv(file_content):
    """Process CSV file content
    
    Args:
        file_content (bytes): The content of the CSV file
        
    Returns:
        dict: Extracted data from the CSV file
    """
    
    # Read CSV content
    df = pd.read_csv(io.BytesIO(file_content))
    
    # Return as dictionary
    return df.to_dict('records')

def process_txt(file_content):
    """Process TXT file content
    
    Args:
        file_content (bytes): The content of the TXT file
        
    Returns:
        dict: Extracted data from the TXT file
    """
    
    # Decode content
    text = file_content.decode('utf-8')
    
    # Try to extract medical data using regex patterns
    data = {}
    
    # Common medical parameters and their regex patterns
    patterns = {
        'glucose': r'glucose.*?(\d+\.?\d*)',
        'blood_pressure': r'blood\s*pressure.*?(\d+\.?\d*)',
        'skin_thickness': r'skin\s*thickness.*?(\d+\.?\d*)',
        'insulin': r'insulin.*?(\d+\.?\d*)',
        'bmi': r'bmi.*?(\d+\.?\d*)',
        'diabetes_pedigree': r'diabetes\s*pedigree.*?(\d+\.?\d*)',
        'age': r'age.*?(\d+)'
    }
    
    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                data[key] = float(match.group(1))
            except ValueError:
                pass
    
    return data if data else {"raw_text": text}

def process_pdf(file_content):
    """Process PDF file content
    
    Args:
        file_content (bytes): The content of the PDF file
        
    Returns:
        dict: Extracted data from the PDF file
    """
    
    # Extract text from PDF
    text = ""
    
    try:
        # Try using PyMuPDF (fitz) first
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
    except:
        # Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        except:
            st.error("Failed to extract text from PDF.")
            return None
    
    # Try to extract medical data using the same approach as TXT files
    data = {}
    
    # Common medical parameters and their regex patterns
    patterns = {
        'glucose': r'glucose.*?(\d+\.?\d*)',
        'blood_pressure': r'blood\s*pressure.*?(\d+\.?\d*)',
        'skin_thickness': r'skin\s*thickness.*?(\d+\.?\d*)',
        'insulin': r'insulin.*?(\d+\.?\d*)',
        'bmi': r'bmi.*?(\d+\.?\d*)',
        'diabetes_pedigree': r'diabetes\s*pedigree.*?(\d+\.?\d*)',
        'age': r'age.*?(\d+)'
    }
    
    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                data[key] = float(match.group(1))
            except ValueError:
                pass
    
    return data if data else {"raw_text": text}

def process_docx(file_content):
    """Process DOCX file content
    
    Args:
        file_content (bytes): The content of the DOCX file
        
    Returns:
        dict: Extracted data from the DOCX file
    """
    
    # Extract text from DOCX
    text = docx2txt.process(io.BytesIO(file_content))
    
    # Try to extract medical data using the same approach as TXT files
    data = {}
    
    # Common medical parameters and their regex patterns
    patterns = {
        'glucose': r'glucose.*?(\d+\.?\d*)',
        'blood_pressure': r'blood\s*pressure.*?(\d+\.?\d*)',
        'skin_thickness': r'skin\s*thickness.*?(\d+\.?\d*)',
        'insulin': r'insulin.*?(\d+\.?\d*)',
        'bmi': r'bmi.*?(\d+\.?\d*)',
        'diabetes_pedigree': r'diabetes\s*pedigree.*?(\d+\.?\d*)',
        'age': r'age.*?(\d+)'
    }
    
    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                data[key] = float(match.group(1))
            except ValueError:
                pass
    
    return data if data else {"raw_text": text}

def process_json(file_content):
    """Process JSON file content
    
    Args:
        file_content (bytes): The content of the JSON file
        
    Returns:
        dict: Extracted data from the JSON file
    """
    
    try:
        # Parse JSON content
        data = json.loads(file_content.decode('utf-8'))
        return data
    except json.JSONDecodeError:
        st.error("Invalid JSON format.")
        return None
