import os
import tempfile
import json
import re
import pytesseract

from openai import OpenAI
import streamlit as st
from pdf2image import convert_from_path
from pytesseract import Output
from docx2pdf import convert


st.set_page_config(
    page_title="gpt-4",
    page_icon="🤖",
    initial_sidebar_state="expanded",
    # layout="wide"
)

client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)


sections_list = ["personal", "contact", "summary", "education", "experience", "skills"]


# Function to read PDF files using tempfile
def read_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        images = convert_from_path(tmp_file.name)
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
    sections = []
    current_section = []
    previous_top = 0

    for page in ocr_results:
        n_boxes = len(page['text'])
        for i in range(n_boxes):
            text = page['text'][i].strip()
            if not text:
                continue

            left = page['left'][i]
            top = page['top'][i]
            width = page['width'][i]
            height = page['height'][i]

            if top - previous_top > height * 1.5:  # Simple heuristic to detect new sections based on spacing
                if current_section:
                    sections.append(current_section)
                current_section = []

            current_section.append((text, left, top, width, height))
            previous_top = top

        if current_section:
            sections.append(current_section)
            current_section = []

    with st.expander("OCR position data", expanded=False):
        st.info(f"{sections}")
    return sections


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
# def create_sections_dict(cleaned_text):
#     sections_dict = {}
#     for line_idx in range(len(cleaned_text)):
#         current_section = str(line_idx)
#         sections_dict[current_section] = []
#         sections_dict[current_section].append(cleaned_text[line_idx])
#
#     return sections_dict


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


# Function to provide few-shot examples based on the section
def get_few_shot_examples(examples, section):
    section_examples = [example for example in examples[section] if example["output"]["section"] == section]
    formatted_examples = ""
    for example in section_examples:
        formatted_examples += f"Input: {example['input']}\n\nOutput: {json.dumps(example['output'], indent=2)}\n\n"
    # st.info(f"get_few_shot_examples output: {formatted_examples}")
    return formatted_examples


# Function to parse text with GPT-4 Medium for a specific section
def parse_text_with_gpt4(text, few_shot_examples):
    prompt = [{"role": "system", "content": f"""
        You are a helpful assistant who extracts information from a resume in the below example JSON format.
        Make sure to extract as much information as present in the document and provide them in the corresponding sections of the resume as per the structure provided in JSON examples.

        JSON Example:
        {few_shot_examples}
        """}, {"role": "user", "content": f"""{text}"""}]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=prompt
    )

    return response.choices[0].message.content


# Streamlit app
st.title("Resume Parser")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

# Load the few-shot examples
with open('main/few_shot_examples_gpt4.json', 'r') as file:
    few_shot_examples = json.load(file)

if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            images = read_pdf(uploaded_file)
            ocr_results = ocr_images(images)
            text_lines = extract_text_with_bboxes(ocr_results)
            # st.info("debug 1")
            # st.info(f"extract_text_with_bboxes output: {text_lines}")
            extracted_text = [" ".join([line[0] for line in lines]) for lines in text_lines]
            with st.expander("OCR extracted text", expanded=False):
                st.info(f"{extracted_text}")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = read_docx(uploaded_file)
        else:
            st.error("Unsupported file type")

        if extracted_text:
            # st.info(f"Extracted text: {extracted_text}")
            clean_text_data = clean_text(extracted_text)
            # st.info(f"Cleaned text: {clean_text_data}")
            sections = create_sections_dict(clean_text_data)
            with st.expander("Section wise text", expanded=True):
                st.json(sections)
            parsed_data = {}

            # Process each section one by one
            for section, section_text in sections.items():
                # st.info(f"Parsing section {section}")
                section_text = "\n\n".join(section_text)
                # with st.expander(f"Cleaned text for section {section}", expanded=False):
                #     st.info(f"{section_text}")
                # entries = split_section_entries(clean_section_text)
                # st.info(f"Entries: {entries}")
                # entry = "\n\n".join(section_text)
                # with st.expander(f"Model input for section {section}", expanded=False):
                #     st.info(f"{section_text}")
                parsed_response = parse_text_with_gpt4(section_text, few_shot_examples)
                # st.markdown(parsed_response)
                response_dict = json.loads(parsed_response)

                for sec, value in response_dict.items():
                    if sec not in parsed_data.keys():
                        parsed_data[sec] = []
                    parsed_data[sec].append(value) # Use extend instead of append
                    # st.info(f"New entry added to section {sec}")

            st.success("Resume parsed successfully!")
            st.json(parsed_data)
    except Exception as e:
        st.info(f"An error occurred: {e}")


# Function to read DOCX files
def read_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        # Convert DOCX to PDF using docx2pdf
        pdf_path = tmp_file.name.replace('.docx', '.pdf')
        convert(tmp_file.name, pdf_path)
        # Convert PDF to images using pdf2image
        images = convert_from_path(pdf_path)
    return images
