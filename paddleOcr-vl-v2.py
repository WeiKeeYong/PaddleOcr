import base64
import mimetypes
import os
from openai import OpenAI

# Function to encode a file to base64
def encode_file_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type if mime_type else "application/octet-stream"

# --- Configuration ---
# Path to your local file (PDF, JPG, BMP, CSV, TXT)
file_path = "D:\\temp\\240130-understanding-wealth-and-personal-banking-transcript.pdf"  # <<< CHANGE THIS TO YOUR FILE PATH

client = OpenAI(
    api_key="EMPTY",
    base_url="http://34.186.117.48:8000/v1",
    timeout=3600
)

# Task-specific base prompts for vision tasks
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
# Choose the task for image-based inputs
CURRENT_TASK = TASKS["ocr"]

# --- File Processing and Message Creation ---

file_extension = os.path.splitext(file_path)[1].lower()
mime_type = get_mime_type(file_path)

messages = [{"role": "user", "content": []}]

# Handle images by encoding them
if file_extension in ['.png', '.jpg', '.jpeg', '.bmp'] or mime_type.startswith('image/'):
    base64_content = encode_file_base64(file_path)
    messages[0]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_content}"
        }
    })
    messages[0]["content"].append({
        "type": "text",
        "text": CURRENT_TASK
    })
elif file_extension == '.pdf':
    # Note: The model may not support direct PDF file inputs.
    # A more robust solution would be to convert each PDF page to an image
    # and send those images. This implementation sends the base64-encoded PDF.
    base64_content = encode_file_base64(file_path)
    messages[0]["content"].append({
        "type": "text",
        "text": f"This is a base64 encoded PDF. Please perform OCR on its content. Content: {base64_content}"
    })

# Handle text-based files by reading their content
elif file_extension in ['.txt', '.csv']:
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            text_content = f.read()
        messages[0]["content"].append({
            "type": "text",
            "text": f"Please process the following text content from the file '{os.path.basename(file_path)}':\n\n{text_content}"
        })
    except Exception as e:
        print(f"Error reading text file: {e}")
        exit()
else:
    print(f"Unsupported file type: {file_extension}. This script supports PNG, JPG, BMP, PDF, TXT, and CSV.")
    exit()


# --- API Call ---
if messages[0]["content"]:
    try:
        response = client.chat.completions.create(
            model="PaddlePaddle/PaddleOCR-VL",
            messages=messages,
            temperature=0.0,
        )
        generated_text = response.choices[0].message.content
        print(f"Generated text: {generated_text}")
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(generated_text)
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
else:
    print("No content was prepared to be sent.")