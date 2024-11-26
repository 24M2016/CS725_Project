import fitz  # PyMuPDF for PDF handling
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from Levenshtein import distance as levenshtein_distance
from PIL import Image
from docx import Document
from pdf2docx import Converter
from sklearn.metrics import precision_score, recall_score, f1_score

# Specify the Tesseract executable path (update as necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# Function to Convert PDF to Word
def pdf_to_word(pdf_path, word_path):
    cv = Converter(pdf_path)
    cv.convert(word_path, start=0, end=None)  # Converts all pages
    cv.close()


# Function to Extract Text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text.strip()


# Function to Extract Text Using OCR
def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text.strip()


# Function to Calculate Metrics (Word-Level Accuracy and Edit Distance)
def calculate_metrics(original_text, generated_text):

    original_tokens = original_text.split()
    generated_tokens = generated_text.split()

    # Word-level Accuracy (common words / total unique words)
    common_tokens = set(original_tokens) & set(generated_tokens)
    word_accuracy = len(common_tokens) / len(set(original_tokens)) if original_tokens else 0

    edit_distance = levenshtein_distance(original_text, generated_text)

    return word_accuracy, edit_distance


# Function to Calculate Character-Level Precision, Recall, and F1-Score
def calculate_character_metrics(original_text, generated_text):

    original_chars = list(original_text)
    generated_chars = list(generated_text)

    # Map characters to binary presence/absence for precision/recall calculation
    unique_chars = list(set(original_chars + generated_chars))
    original_binary = [1 if char in original_chars else 0 for char in unique_chars]
    generated_binary = [1 if char in generated_chars else 0 for char in unique_chars]

    precision = precision_score(original_binary, generated_binary, zero_division=0)
    recall = recall_score(original_binary, generated_binary, zero_division=0)
    f1 = f1_score(original_binary, generated_binary, zero_division=0)

    return precision, recall, f1


# Function to Plot Word Accuracy
def plot_word_accuracy(ocr_metrics, converted_metrics):

    labels = ['OCR', 'Converted']
    word_accuracy = [ocr_metrics[0], converted_metrics[0]]  # Word accuracy

    x = np.arange(len(labels))  # label locations

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Word Accuracy
    ax.bar(x, word_accuracy, width=0.5, color='skyblue', edgecolor='black', label='Word Accuracy')

    ax.set_title('Word Accuracy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Word Accuracy', fontsize=12)
    ax.set_xlabel('Methods', fontsize=12)
    ax.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Function to Plot Edit Distance
def plot_edit_distance(ocr_metrics, converted_metrics):
    labels = ['OCR', 'Converted']
    edit_distance = [ocr_metrics[1], converted_metrics[1]]  # Edit distance

    x = np.arange(len(labels))  # label locations

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Edit Distance
    ax.bar(x, edit_distance, width=0.5, color='salmon', edgecolor='black', label='Edit Distance')

    ax.set_title('Edit Distance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Edit Distance', fontsize=12)
    ax.set_xlabel('Methods', fontsize=12)
    ax.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Function to Plot Character-Level Metrics Comparison
def plot_character_metrics_comparison(ocr_character_metrics, converted_character_metrics):
    labels = ['Precision', 'Recall', 'F1-Score']
    ocr_values = [ocr_character_metrics[0], ocr_character_metrics[1], ocr_character_metrics[2]]
    converted_values = [converted_character_metrics[0], converted_character_metrics[1], converted_character_metrics[2]]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Precision, Recall, F1 Score
    ax.bar(x - width / 2, ocr_values, width, label='OCR', color='lightgreen', edgecolor='black')
    ax.bar(x + width / 2, converted_values, width, label='Converted', color='lightcoral', edgecolor='black')

    ax.set_title('Character-Level Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Main Workflow to Compare OCR vs. Converted Text
def compare_pdf_vs_original_docx(pdf_path, original_docx_path, converted_docx_path):
   # Convert PDF to Word (DOCX)
    pdf_to_word(pdf_path, converted_docx_path)

    # Extract text from DOCX files (original and converted) and using OCR
    original_text = extract_text_from_docx(original_docx_path)
    converted_text = extract_text_from_docx(converted_docx_path)
    ocr_text = extract_text_with_ocr(pdf_path)

    # Calculate Metrics (Word Accuracy and Edit Distance)
    ocr_metrics = calculate_metrics(original_text, ocr_text)
    converted_metrics = calculate_metrics(original_text, converted_text)

    # Calculate Character-Level Metrics (Precision, Recall, F1)
    ocr_character_metrics = calculate_character_metrics(original_text, ocr_text)
    converted_character_metrics = calculate_character_metrics(original_text, converted_text)

    # Print Metrics
    print(f"OCR Word Accuracy: {ocr_metrics[0]:.4f}, Edit Distance: {ocr_metrics[1]}")
    print(f"Converted Word Accuracy: {converted_metrics[0]:.4f}, Edit Distance: {converted_metrics[1]}")

    print(f"OCR Precision: {ocr_character_metrics[0]:.4f}, Recall: {ocr_character_metrics[1]:.4f}, F1: {ocr_character_metrics[2]:.4f}")
    print(f"Converted Precision: {converted_character_metrics[0]:.4f}, Recall: {converted_character_metrics[1]:.4f}, F1: {converted_character_metrics[2]:.4f}")

    # Plot Word Accuracy (Separate Plot)
    plot_word_accuracy(ocr_metrics, converted_metrics)

    # Plot Edit Distance (Separate Plot)
    plot_edit_distance(ocr_metrics, converted_metrics)

    # Plot Character-Level Metrics (Precision, Recall, F1)
    plot_character_metrics_comparison(ocr_character_metrics, converted_character_metrics)


# Example Usage
if __name__ == "__main__":
    try:
        pdf_path = "DocumentsAndPdf_used/1.pdf"  # Path to the input PDF file
        original_docx_path = "DocumentsAndPdf_used/original.docx"  # Path to the original DOCX file
        converted_docx_path = "DocumentsAndPdf_used/converted.pdf"  # Path to the DOCX converted from PDF

        compare_pdf_vs_original_docx(pdf_path, original_docx_path, converted_docx_path)

    except Exception as e:
        print(f"Error: {e}")
