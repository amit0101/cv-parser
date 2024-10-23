import os
import tempfile
import json
import re

import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from pytesseract import Output
from PIL import Image
import openai
import subprocess

# Set the TOKENIZERS_PARALLELISM environment variable to false
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the sections
sections_list = ["personal", "contact", "summary", "education", "experience", "skills"]

# Function to read PDF files using tempfile
def read_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        images = convert_from_path(tmp_file.name)
    return images

def resize_image(img, max_width=1024, max_height=1024):
    img.thumbnail((max_width, max_height), Image.ANTIALIAS)
    return img

# Function to read DOCX files
# Function to read DOCX files and convert to PDF, then to images
def read_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        pdf_path = tmp_file.name.replace(".docx", ".pdf")
        try:
            # Convert DOCX to PDF using libreoffice
            subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", tmp_file.name, "--outdir", os.path.dirname(tmp_file.name)], check=True)
            images = convert_from_path(pdf_path, dpi=150)
            images = [resize_image(img) for img in images]
        except Exception as e:
            st.error("Failed to convert DOCX to PDF. Is LibreOffice installed?")
            raise e
    return images

# Function to perform OCR and get text with bounding boxes
def ocr_images(images):
    ocr_results = []
    for img in images:
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        ocr_results.append(data)
    return ocr_results

# Function to extract text from OCR results using bounding boxes
def extract_text_with_bboxes(ocr_results):
    sections = {}
    current_section = None

    for page_idx, page in enumerate(ocr_results):
        n_boxes = len(page['text'])
        for i in range(n_boxes):
            text = page['text'][i].strip()
            if text:
                # Use spatial data to infer structure
                x, y, w, h = page['left'][i], page['top'][i], page['width'][i], page['height'][i]
                # Assuming section headers are typically in bold and have larger font size
                if page['conf'][i] > 60 and w > 100:  # Confidence threshold and width heuristic
                    normalized_text = re.sub(r'\W+', '', text.lower())
                    if normalized_text in sections_list:
                        current_section = normalized_text
                        sections[current_section] = ""
                        st.write(f"New section detected: {current_section} at position ({x}, {y})")
                if current_section:
                    sections[current_section] += " " + text
    return sections

# Function to identify the section type using GPT-4 Turbo
def identify_section(text):
    prompt = (f"Identify the section type for the input text following the example. The identified section should strictly be from the list: {', '.join(sections_list)}.\n\n" +
              "Example:\nInput: MSc and BSc in Mathematics\nSection: education\n\n" +
              f"Input: {text}\nSection:")

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    st.write(f"Response: {response}")
    section = response['choices'][0]['message']['content'].strip()
    st.write(f"Identified section for text: {text}\nSection: {section}")
    return section

def get_few_shot_examples(examples, section):
    section_examples = [example for example in examples if example["output"]["section"] == section]
    formatted_examples = ""
    for example in section_examples:
        formatted_examples += f"Input: {example['input']}\nOutput: {json.dumps(example['output'], indent=2)}\n\n"
    return formatted_examples

# Function to parse text with GPT-4 Turbo for a specific section
def parse_text_with_llm_for_section(text, section, examples):
    few_shot_examples = get_few_shot_examples(examples, section)
    prompt = f"Extract the {section} information from the following text and provide it only in JSON format. Ensure the JSON structure matches the examples provided.\n\n{few_shot_examples}\nInput: {text}\nOutput:"

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )

    st.write(f"Response: {response}")
    response_text = response['choices'][0]['message']['content'].strip()
    st.info(f"Model Response for section {section}: {response_text}")

    # Extract JSON from the response
    # json_output = extract_json_from_response(response_text)
    return response_text

# Function to clean the extracted text
def clean_text(text):
    clean_text = text.replace("\n", " ").replace("\t", " ").strip()
    return clean_text

# Function to extract JSON from the model's response
def extract_json_from_response(response):
    try:
        start_idx = response.rindex("{")
        end_idx = response.rindex("}") + 1
        json_response = response[start_idx:end_idx]
        return json.loads(json_response)
    except (ValueError, json.JSONDecodeError) as e:
        st.write(f"Error extracting JSON: {e}")
        st.write(f"Response received: {response}")
        return {}

# Function to split section text into individual entries
def split_section_entries(text):
    # Implement a method to split the section text into individual entries
    # This can be based on newlines, bullet points, or any other delimiter
    return re.split(r'(\*|\n\n|\n)', text)  # Example split based on double newlines or asterisks

# Streamlit app
st.title("Resume Parser")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

# Load the few-shot examples
with open('main/few_shot_examples.json', 'r') as file:
    few_shot_examples = json.load(file)

if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            images = read_pdf(uploaded_file)
            ocr_results = ocr_images(images)
            sections = extract_text_with_bboxes(ocr_results)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            ocr_results = read_docx(uploaded_file)
            sections = extract_text_with_bboxes(ocr_results)
        else:
            st.error("Unsupported file type")

        if sections:
            st.info(f"Extracted sections: {sections}")
            parsed_data = {}

            # Process each section one by one
            for section, text in sections.items():
                st.info(f"Parsing section: {section}")
                clean_section_text = clean_text(text)
                entries = split_section_entries(clean_section_text)
                for entry in entries:
                    st.info(f"Processing entry: {entry}")
                    identified_section = identify_section(entry)
                    st.info(f"Identified section: {identified_section}")
                    parsed_entry = parse_text_with_llm_for_section(entry, identified_section, few_shot_examples)
                    st.info("Parsed entry", parsed_entry)
                    if identified_section not in parsed_data:
                        parsed_data[identified_section] = []
                    parsed_data[identified_section].extend(parsed_entry)  # Use extend instead of append

            st.success("Resume parsed successfully!")
            st.json(parsed_data)
    except Exception as e:
        st.error(f"An error occurred: {e}")
