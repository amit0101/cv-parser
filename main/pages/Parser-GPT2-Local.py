import os
import tempfile
import json
import torch
import re
import pytesseract

import streamlit as st
from pdf2image import convert_from_path
from pytesseract import Output
from docx2pdf import convert
import fitz

from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(
    page_title="gpt-2",
    page_icon="🤖",
    initial_sidebar_state="expanded",
    # layout="wide"
)

# Set the TOKENIZERS_PARALLELISM environment variable to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sections_list = ["personal", "contact", "summary", "education", "experience", "skills"]

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize the model
model_name = "gpt2-medium"  # Using GPT-2 Medium
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Function to read PDF files using tempfile
def read_pdf(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file.flush()
            pdf_path = tmp_file.name

        pdf_document = fitz.open(pdf_path)
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

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
def create_sections(cleaned_text):
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


# Function to parse text with GPT-2 Medium for a specific section
def parse_text_with_gpt2_for_section(text, section, model, tokenizer, examples, max_length=256, max_new_tokens=50):
    few_shot_examples = get_few_shot_examples(examples, section)
    prompt = f"You are a helpful assistant who extracts the {section} information from the last Input in the following text and provides it only in JSON format. Ensure the JSON response for Output has the same keys as the examples provided but don't return any values from the example outputs. Leave the values as empty text that you can't extract.\n\nExamples: {few_shot_examples}\n\nInput: {text}\n\nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs.input_ids

    # Chunk the input IDs if they exceed the max length
    chunks = [input_ids[0][i:i + max_length] for i in range(0, len(input_ids[0]), max_length)]
    # st.info("Chunks generated")

    all_outputs = []

    for chunk in chunks:
        chunk_input_ids = chunk.unsqueeze(0)
        attention_mask = torch.ones_like(chunk_input_ids).to(device)

        outputs = model.generate(
            chunk_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Debugging: Print the model's response
        with st.expander(f"Model Response for section {section}", expanded=False):
            st.info(f"{response}")

        # Extract JSON from the response
        json_output = extract_json_from_response(response)

        if isinstance(json_output, dict):
            with st.expander(f"Parsed response for {section}", expanded=False):
                st.json(json_output)
            all_outputs.append(json_output)
        else:
            st.info(f"Unexpected JSON output type: {type(json_output)}")

        # Clear cache to free up memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return all_outputs


# Function to extract JSON from the model's response
def extract_json_from_response(text):
    pattern = re.compile(r'Output:\s*{')
    matches = pattern.finditer(text)

    match_count = 0
    for match in matches:
        start = match.end() - 1
        brace_count = 1
        end = start + 1
        while brace_count > 0 and end < len(text):
            if text[end] == '{':
                brace_count += 1
            elif text[end] == '}':
                brace_count -= 1
            end += 1
        json_object = text[start:end]
        match_count += 1
        if match_count == 2:  # Process the second match
            try:
                parsed_json = json.loads(json_object)
                return parsed_json  # Return the second JSON object and stop
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"JSON string: {json_object}")

    return None  # Return None if the second JSON object is not found


# Function to split section text into individual entries
def split_section_entries(text):
    # Implement a method to split the section text into individual entries
    # This can be based on newlines, bullet points, or any other delimiter
    return text.split("\n\n")  # Example split based on double newlines or asterisks


# Streamlit app
st.title("Resume Parser")

st.info("This implementation of the lightweight gpt-2-medium model is meant for running locally or on a server. It can be very slow if run on this app.")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

# Load the few-shot examples
with open('main/few_shot_examples.json', 'r') as file:
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
            sections = create_sections(clean_text_data)
            with st.expander("Section wise text", expanded=True):
                st.json(sections)
            parsed_data = {}

            # Process each section one by one
            for section, text in sections.items():
                st.info(f"Parsing section: {section}")
                clean_section_text = clean_text(text)
                with st.expander(f"Cleaned section text: {section}", expanded=False):
                    st.info(f"{clean_section_text}")
                # entries = split_section_entries(clean_section_text)
                # st.info(f"Entries: {entries}")
                for entry in clean_section_text:
                    with st.expander(f"Processing entry: {section}", expanded=False):
                        st.info(f"{entry}")
                    # identified_section = identify_section(entry, model, tokenizer)
                    # st.info(f"Identified section: {identified_section}")
                    # parsed_entry = parse_text_with_gpt2_for_section(entry, identified_section, model, tokenizer,
                    #                                                 few_shot_examples)
                    parsed_entry = parse_text_with_gpt2_for_section(entry, section, model, tokenizer,
                                                                    few_shot_examples)
                    if section not in parsed_data:
                        parsed_data[section] = []
                    parsed_data[section].extend(parsed_entry)  # Use extend instead of append
                    st.info(f"New entry added to section {section}")

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

