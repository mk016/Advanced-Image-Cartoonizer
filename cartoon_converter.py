import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as transforms
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Advanced Image Cartoonizer",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

def apply_anime_style(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter with stronger parameters
    color = cv2.bilateralFilter(img, 15, 80, 80)
    
    # Edge detection with more defined lines
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 5)
    
    # Combine color and edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Color enhancement for anime look
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation for vibrant colors
    s = cv2.multiply(s, 1.4)
    # Increase brightness
    v = cv2.add(v, 15)
    
    # Merge channels
    hsv = cv2.merge([h, s, v])
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Final enhancement
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=15)
    
    return cartoon

def apply_ghibli_style(img):
    # Convert to float32
    img_float = img.astype(np.float32) / 255.0
    
    # Enhance edges for character definition
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 2)
    
    # Apply bilateral filter for smooth color regions
    smooth = cv2.bilateralFilter(img, 9, 150, 150)
    
    # Color quantization with more colors for Ghibli palette
    Z = smooth.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12  # Increased number of colors
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape((smooth.shape))
    
    # Adjust color temperature for warmer Ghibli tones
    lab = cv2.cvtColor(quantized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Increase warmth
    a = cv2.add(a, 3)  # Subtle red
    b = cv2.add(b, 8)  # More yellow
    
    lab = cv2.merge([l, a, b])
    warmer = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Edge enhancement for character features
    kernel = np.array([[0, -1, 0],
                      [-1, 5,-1],
                      [0, -1, 0]])
    sharpened = cv2.filter2D(warmer, -1, kernel)
    
    # Soft edge blending
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(sharpened, 0.85, edges_3channel, 0.15, 0)
    
    # Final color adjustments
    hsv = cv2.cvtColor(blend, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.1)  # Slightly increase saturation
    v = cv2.add(v, 5)      # Subtle brightness boost
    
    hsv = cv2.merge([h, s, v])
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return final

def apply_pixar_style(img):
    # Convert to Lab color space for better color manipulation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply bilateral filter for smooth surfaces
    smooth = cv2.bilateralFilter(enhanced, 9, 200, 200)
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 9)
    
    # Combine edges with smooth image
    cartoon = cv2.bitwise_and(smooth, smooth, mask=edges)
    
    # Color enhancement for Pixar look
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation and brightness
    s = cv2.multiply(s, 1.4)
    v = cv2.add(v, 20)
    
    hsv = cv2.merge([h, s, v])
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return cartoon

def apply_sketch_style(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create negative image
    negative = 255 - gray
    
    # Apply GaussianBlur
    blur = cv2.GaussianBlur(negative, (21, 21), 0)
    
    # Blend to create sketch
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    
    # Enhance contrast
    sketch = cv2.convertScaleAbs(sketch, alpha=1.2, beta=10)
    
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_watercolor_style(img):
    # Bilateral filter for smooth regions
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Create edge mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 9, 2)
    
    # Apply median blur for watercolor texture
    median = cv2.medianBlur(smooth, 7)
    
    # Color quantization
    Z = median.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 9
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape((median.shape))
    
    # Soft light effect
    gaussian = cv2.GaussianBlur(quantized, (7, 7), 0)
    watercolor = cv2.addWeighted(quantized, 0.6, gaussian, 0.4, 0)
    
    return watercolor

def apply_comic_style(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create edge mask
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.dilate(edges, None)
    edges = cv2.bitwise_not(edges)
    
    # Color quantization
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape((img.shape))
    
    # Apply bilateral filter
    smooth = cv2.bilateralFilter(quantized, 9, 150, 150)
    
    # Combine with edges
    comic = cv2.bitwise_and(smooth, smooth, mask=edges)
    
    # Enhance contrast
    comic = cv2.convertScaleAbs(comic, alpha=1.2, beta=10)
    
    return comic

def process_image(image, style):
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply selected style
    if style == "Anime":
        result = apply_anime_style(img)
    elif style == "Ghibli":
        result = apply_ghibli_style(img)
    elif style == "Pixar":
        result = apply_pixar_style(img)
    elif style == "Sketch":
        result = apply_sketch_style(img)
    elif style == "Watercolor":
        result = apply_watercolor_style(img)
    elif style == "Comic":
        result = apply_comic_style(img)
    
    # Convert back to RGB for display
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def main():
    st.title("🎨 Advanced Image Cartoonizer")
    st.markdown("""
    Transform your images into different cartoon styles using advanced image processing techniques.
    Choose from multiple styles and get your cartoonized image instantly!
    """)
    
    # Create two columns for the layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.subheader("Choose Style")
        style = st.selectbox(
            "Select cartoon style",
            ["Anime", "Ghibli", "Pixar", "Sketch", "Watercolor", "Comic"],
            help="Choose the style you want to apply to your image"
        )
        
        if uploaded_file is not None:
            if st.button("Generate Cartoon", type="primary"):
                with st.spinner("Creating your cartoon masterpiece... 🎨"):
                    # Process the image
                    cartoon_image = process_image(original_image, style)
                    
                    # Display the result
                    st.image(cartoon_image, caption=f"{style} Style", use_column_width=True)
                    
                    # Add download button
                    st.download_button(
                        label="Download Cartoon Image",
                        data=cartoon_image.tobytes(),
                        file_name=f"cartoon_{style.lower()}.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main() 