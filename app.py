from flask import Flask, request, render_template, send_file, url_for, send_from_directory
from PyPDF2 import PdfReader
from transformers import pipeline
from datasets import Dataset
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import re
import textwrap
import zipfile
from io import BytesIO

app = Flask(__name__)

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Create a folder for storing generated PDFs if it doesn't exist
if not os.path.exists("generated_pdfs"):
    os.makedirs("generated_pdfs")

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=400):
    """Chunk large text into smaller pieces for summarization."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(" ".join(current_chunk + [sentence])) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    return chunks

def summarize_text(text):
    """Summarize the extracted text using dataset for batch processing."""
    chunks = chunk_text(text)

    # Create a dataset from the chunks
    dataset = Dataset.from_dict({"text": chunks})

    # Batch process the summarization
    summaries = summarizer(dataset["text"], max_length=60, min_length=20, do_sample=False)

    # Extract the summarized texts from the results
    summarized_texts = [summary["summary_text"] for summary in summaries]
    
    # Combine the summaries into one final summary
    final_summary = " ".join(summarized_texts)
    
    # If the final summary is shorter than 100 words, we add more summarization (optional)
    if len(final_summary.split()) < 100:
        final_summary += " " + summarizer(final_summary, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]

    return final_summary


def generate_pdf(summary_text, filename="summary.pdf"):
    """Generate a PDF from the summary text with adjusted right margin."""
    
    # Sanitize the filename to avoid invalid characters
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    pdf_file_path = os.path.join("generated_pdfs", filename)
    
    # Create PDF using reportlab
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    width, height = letter

    # Add title to PDF
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 40, "Summary")

    # Set font for the summary text
    c.setFont("Helvetica", 10)

    # Adjust the wrapping to allow a wider right margin
    # Change the width to 80 to extend the right margin (default 90 for more narrow wrapping)
    wrapped_text = textwrap.wrap(summary_text, width=120)  # Reduce the wrap width to increase right margin

    # Set starting y position for the text
    y_position = height - 60

    # Line spacing (distance between each line of text)
    line_height = 12  # Adjust line height to space out text

    # Add text to PDF, taking care of page overflow
    for line in wrapped_text:
        # Check if the text has reached the bottom of the page
        if y_position < 40:
            c.showPage()  # Start a new page
            c.setFont("Helvetica", 10)  # Reset font for the new page
            y_position = height - 40  # Reset position for new page

        c.drawString(30, y_position, line)  # Draw the line of text at the current position
        y_position -= line_height  # Move down for the next line

    # Save the PDF
    c.showPage()
    c.save()
    
    return pdf_file_path


@app.route("/")
def index():
    """Render the main upload page."""
    return render_template("index.html")

@app.route("/summarizeMultipart")
def multipartIndex():
    """Render the Multipart upload page"""
    return render_template("multifile.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        # Check if a file is uploaded
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        
        if file.filename == "":
            return "No selected file", 400
        
        if file:
            # Ensure the uploads directory exists
            if not os.path.exists("uploads"):
                os.makedirs("uploads")

            # Save the file temporarily
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            
            # Extract text from the uploaded PDF
            text = extract_text_from_pdf(file_path)

            # Summarize the extracted text
            summary = summarize_text(text)
            
            # Generate the PDF from the summary
            pdf_file_path = generate_pdf(summary, filename=f"{os.path.splitext(file.filename)[0]}_summary.pdf")
            
            # Optionally, remove the file after processing
            os.remove(file_path)

            # Provide the generated PDF for download
            return render_template("summary.html", summary=summary, pdf_filename=os.path.basename(pdf_file_path))

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route("/summarizeMultipart", methods=["POST"])
def summarizeMultipart():
    try:
        # Check if files are uploaded
        if "files" not in request.files:
            return "No file part", 400

        files = request.files.getlist("files")
        
        if not files:
            return "No selected files", 400
        
        # Process each uploaded PDF
        pdf_paths = []  # List to store paths of generated PDFs
        for file in files:
            if file.filename == "":
                continue
            
            # Save the file temporarily
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            
            # Extract text from the uploaded PDF
            text = extract_text_from_pdf(file_path)

            # Summarize the extracted text
            summary = summarize_text(text)
            
            # Generate the PDF from the summary
            pdf_file_path = generate_pdf(summary, filename=f"{os.path.splitext(file.filename)[0]}_summary.pdf")
            pdf_paths.append(pdf_file_path)
            
            # Optionally, remove the file after processing
            os.remove(file_path)

        # If multiple PDFs are uploaded, create a zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for pdf_path in pdf_paths:
                zip_file.write(pdf_path, os.path.basename(pdf_path))
                os.remove(pdf_path)  # Remove the file after adding to the zip
        
        # Set the position of the buffer to the beginning
        zip_buffer.seek(0)
        
        # Provide the zip file for download
        return send_file(zip_buffer, as_attachment=True, download_name="summarized_pdfs.zip", mimetype="application/zip") 

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route("/download/<filename>")
def download_pdf(filename):
    """Serve the generated PDF for download."""
    return send_from_directory("generated_pdfs", filename)

if __name__ == '__main__':
    app.run(debug=True)
