import base64
from openai import OpenAI

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your local image
data_path = "D:\\temp\\receipt.png"

# Getting the base64 string
base64_data = encode_image(data_path)

client = OpenAI(
    api_key="EMPTY",
    base_url="http://34.186.117.48:8000/v1",
    timeout=3600
)

# Task-specific base prompts
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_data}"
                }
            },
            {
                "type": "text",
                "text": TASKS["ocr"]
            }
        ]
    }
]

response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
)
print(f"Generated text: {response.choices[0].message.content}")