import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time
import os

# Set up Streamlit app configurations
st.set_page_config(
    page_title="Advanced Image Classification & Object Detection",
    page_icon="📷",
    layout="wide"
)

# Load pre-trained model paths
MODEL_PATH = "C:/Users/Basha/OneDrive/Desktop/my git hub app/tnsermodel/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
model = tf.saved_model.load(MODEL_PATH)

# Sidebar for options
st.sidebar.title("Settings ⚙️")
st.sidebar.markdown("Customize your object detection experience!")

# 1. Detection Confidence Threshold Slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, 0.05, help="Adjust confidence level to filter object detections"
)

# 2. Class Selection (Filter)
classes_to_detect = st.sidebar.multiselect(
    "Classes to Detect",
    options=[f"Class {i}" for i in range(1, 91)],  # Example, can be customized based on model classes
    default=["Class 1", "Class 2"]
)

# 3. User Input for Searching Specific Object
search_object = st.sidebar.text_input(
    "Search for Specific Object", help="Enter the object name or class you want to detect."
)

# 4. Upload Images or Video
uploaded_files = st.sidebar.file_uploader(
    "Upload Images/Video", type=["jpg", "jpeg", "png", "mp4"], accept_multiple_files=True
)
batch_mode = st.sidebar.checkbox("Batch Mode (for multiple images)", False)

# Main Title
st.markdown("<h1 style='text-align: center;'>🔍 Advanced Image Classification & Object Detection</h1>", unsafe_allow_html=True)
st.write("A powerful tool for detecting objects in images and videos using deep learning. Customize settings, batch process images, and download results!")

# Function to run object detection on an image
def detect_objects(image):
    # Convert image to RGB (3 channels) if it's not already
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    
    # Check if image has an alpha channel (4 channels)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Remove alpha channel and convert to RGB

    # Resize image to fit model's expected input (320x320)
    image = cv2.resize(image, (320, 320))  # Adjust the size based on model's input requirement

    # Prepare image for model (convert to tensor and add batch dimension)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run inference
    detections = model(input_tensor)

    # Process detection results
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    return detection_boxes, detection_classes, detection_scores

# Display function for single or batch image detection
def display_results(image, boxes, classes, scores, threshold=0.5, search_object=None):
    height, width, _ = image.shape
    annotated_image = image.copy()

    # If the user has entered a search object, we only highlight the matching object
    search_found = False
    
    for i in range(len(boxes)):
        if scores[i] >= threshold:
            detected_class = f"Class {classes[i]}"
            if search_object:
                # Check if the detected object matches the search input (case insensitive)
                if search_object.lower() in detected_class.lower():
                    search_found = True
                    box = boxes[i]
                    start_point = (int(box[1] * width), int(box[0] * height))
                    end_point = (int(box[3] * width), int(box[2] * height))
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_image, start_point, end_point, color, 2)
                    label = f"{detected_class}: {scores[i]:.2f}"
                    cv2.putText(annotated_image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # If no search query, show all detected objects
                box = boxes[i]
                start_point = (int(box[1] * width), int(box[0] * height))
                end_point = (int(box[3] * width), int(box[2] * height))
                color = (0, 255, 0)
                cv2.rectangle(annotated_image, start_point, end_point, color, 2)
                label = f"Class {classes[i]}: {scores[i]:.2f}"
                cv2.putText(annotated_image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if search_object and not search_found:
        st.warning(f"No matching object found for: {search_object}")
    
    return annotated_image

# Real-time Video Detection (if video is uploaded)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Run object detection
            boxes, classes, scores = detect_objects(image)
            annotated_image = display_results(image, boxes, classes, scores, threshold=confidence_threshold, search_object=search_object)
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        elif uploaded_file.type == "video/mp4":
            st.video(uploaded_file)

            # Optionally, add real-time object detection on video frames here
            st.write("Real-time object detection on video is in development.")

# Download Options
if st.button("Download Results as CSV"):
    # Create DataFrame of detections
    results = pd.DataFrame({
        "Class": [f"Class {cls}" for cls in classes],
        "Score": [f"{score:.2f}" for score in scores]
    })
    st.download_button(label="Download Detection Results", data=results.to_csv(index=False), file_name="detection_results.csv", mime="text/csv")

# Sidebar User Guide
st.sidebar.markdown("### 💡 How to Use")
st.sidebar.markdown(
    """
    1. **Upload Image/Video**: Choose single or multiple files.
    2. **Customize Settings**: Adjust threshold, classes, and batch mode.
    3. **Search Object**: Enter the name or class of the object you want to find in the uploaded file.
    4. **Download Results**: Export detection data as CSV.
    """
)
st.sidebar.info("Enjoy advanced image analysis with our intuitive tool!")
