# Advanced Image Cartoonizer

This project converts regular images into various cartoon styles using advanced image processing techniques. It provides a simple web interface using Streamlit where users can upload images and get their cartoon versions in different styles.

## Features

- Upload images (JPG, JPEG, PNG)
- Convert images to multiple cartoon styles:
  - Anime
  - Ghibli
  - Pixar
  - Sketch
- Download the cartoon version
- Simple and intuitive web interface
- Loading spinner during processing
- Styled interface with responsive layout

## Installation

1. Clone this repository

2. Create and activate a virtual environment (recommended):
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

3. Upgrade pip and install dependencies:
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install required packages
pip3 install -r requirements.txt
```

## Usage

1. Make sure your virtual environment is activated:
```bash
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

2. Run the Streamlit app:
```bash
python3 -m streamlit run cartoon_converter.py
```

3. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)
4. Upload an image using the file uploader
5. Select a cartoon style from the dropdown
6. Click "Generate Cartoon" to create the cartoon version
7. Download the result using the download button

## Troubleshooting

If you encounter any issues:

1. Make sure your virtual environment is activated (you should see `(venv)` in your terminal)
2. Try reinstalling the requirements:
```bash
pip3 install --upgrade -r requirements.txt
```
3. If Streamlit command is not found, use:
```bash
python3 -m streamlit run cartoon_converter.py
```

## How it works

The cartoon effect is achieved through the following steps:
1. Convert the image to grayscale for edge detection
2. Apply bilateral filter for smooth color regions
3. Use adaptive thresholding for edge detection
4. Apply color quantization for distinct color blocks
5. Adjust color temperature for warmth
6. Enhance edges for character features
7. Blend edges with color for a natural look
8. Adjust saturation and brightness for the final effect

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Streamlit
- Pillow
- scikit-image
- Torch
- Torchvision

## Enhancements

- Custom CSS for better styling
- Responsive layout with proper image scaling
- Emoji icons for better visual appeal
- Help text for style selection

This project is inspired by the distinctive art styles of Studio Ghibli, Pixar, and other animation studios, providing a fun and creative way to transform your images into cartoon masterpieces. # ML-Project2
