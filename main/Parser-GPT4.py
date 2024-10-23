import os
import tempfile
import json
import re
from PIL import Image
import streamlit as st
from pdf2image import convert_from_path
import ocrmypdf
import cv2
import numpy as np

# st.set_page_config(
#     page_title="CV Parser",
#     page_icon="🤖",
#     initial_sidebar_state="expanded",
# )

# Sections to be identified in the resume
sections_list = ["personal", "contact", "summary", "education", "experience", "skills"]

# Function to read PDF files using tempfile
def read_pdf(file, max_pages=3):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file.flush()
            pdf_path = tmp_file.name

        # Run OCR on the PDF to extract text
        ocrmypdf.ocr(pdf_path, pdf_path, use_threads=True)

        pdf_document = fitz.open(pdf_path)
        images = []

        for page_num in range(min(len(pdf_document), max_pages)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = resize_image(img)
            images.append(img)

        return images

# Function to read DOCX files
def read_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        # Convert DOCX to images using pdf2image
        images = convert_from_path(tmp_file.name, dpi=150)
        # Convert PDF to images using pdf2image
        images = convert_from_path(pdf_path, dpi=150)
        images = [resize_image(img) for img in images]
    return images

# Function to resize images to reduce memory usage
def resize_image(img, max_width=1024, max_height=1024):
    img.thumbnail((max_width, max_height), Image.ANTIALIAS)
    return img

# Function to preprocess images
def preprocess_image(img):
    # Convert image to grayscale
    img = img.convert('L')
    # Convert to numpy array for further processing
    img_np = np.array(img)
    # Apply thresholding for binarization
    _, img_np = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Convert back to PIL image
    img = Image.fromarray(img_np)
    return img


# Function to clean the extracted text
def clean_text(text_lines):
    cleaned_lines = []
    combined_text = ""

    for i, text in enumerate(text_lines):
        text = re.sub(r'[^A-Za-z0-9/*@ ,.-]', '', text)  # Remove special characters except comma, dot, and hyphen

        if combined_text:
            combined_text += " " + text
        else:
            combined_text = text

        if text.endswith(","):
            continue  # Continue combining if the line ends with a comma

        # Check if the next element starts with a lowercase letter
        if i + 1 < len(text_lines) and text_lines[i + 1][0].islower():
            continue  # Continue combining if the next line starts with a lowercase letter

        cleaned_lines.append(combined_text.strip())
        combined_text = ""

    if combined_text:
        cleaned_lines.append(combined_text.strip())

    return cleaned_lines

# Function to create sections from the cleaned text
def create_sections_dict(cleaned_text):
    sections = {}
    current_section = "personal"
    sections[current_section] = []

    # Define regex patterns for detecting section headers
    section_patterns = {
        "personal": re.compile(r'^\s*personal\s*$', re.I),
        "contact": re.compile(r'^\s*contact\s*$', re.I),
        "summary": re.compile(r'^\s*summary\s*$', re.I),
        "education": re.compile(r'^\s*education\s*$', re.I),
        "experience": re.compile(r'^\s*experience\s*$', re.I),
        "languages": re.compile(r'^\s*languages\s*$', re.I),
        "skills": re.compile(r'^\s*skills\s*$', re.I),
        "achievements": re.compile(r'^\s*achievements\s*$', re.I),
        "certifications": re.compile(r'^\s*certifications\s*$', re.I),
        "qualifications": re.compile(r'^\s*qualifications\s*$', re.I)
    }

    for line in cleaned_text:
        section_identified = False
        for section, pattern in section_patterns.items():
            if pattern.search(line):
                current_section = section
                sections[current_section] = []
                section_identified = True
                st.info(f"Section identified: {section} ")
                break

        if not section_identified:
            sections[current_section].append(line)

    return sections

# Streamlit app
st.title("Resume Parser")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            images = read_pdf(uploaded_file)
            ocr_results = []  # Placeholder for OCR results
            extracted_text = [text for text in ocr_results]
            with st.expander("OCR extracted text", expanded=False):
                st.info(f"{extracted_text}")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            images = read_docx(uploaded_file)
            ocr_results = ocr_images_pytesseract(images)
            extracted_text = [text for text in ocr_results]
        else:
            st.error("Unsupported file type")

        if extracted_text:
            clean_text_data = clean_text(extracted_text)
            sections = create_sections_dict(clean_text_data)
            with st.expander("Section wise text", expanded=True):
                st.json(sections)
    except Exception as e:
        st.error(f"An error occurred: {e}")
