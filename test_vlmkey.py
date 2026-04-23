import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import pymupdf

load_dotenv()

# Install these first if you haven't:
# pip install pymupdf Pillow google-genai

client = genai.Client(api_key=os.getenv("VLM_GEMINI_API_KEY"))

# Open the PDF and render page 4 as an image
# Page 4 of Attention Is All You Need has the attention formula
doc = pymupdf.open(r"data\Attention is all you need.pdf")
page = doc[3]  # 0-indexed so page 4 = index 3
mat = pymupdf.Matrix(2, 2)  # 2x zoom for clarity
pix = page.get_pixmap(matrix=mat)
pix.save("data/test_page.png")

# Send it to Gemini Vision
image = Image.open("data/test_page.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        image,
        "List all mathematical formulas you can see on this page. Write each one in plain text."
    ]
)

print(response.text)