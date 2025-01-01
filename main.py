import streamlit as st
import cv2 as cv
import numpy as np
from typing import List, Tuple
import tempfile
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

def scale_image_for_display(image, max_width=1200):  # Increased max_width
    """Scale image to fit display while maintaining aspect ratio"""
    height, width = image.shape[:2]
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv.resize(image, (new_width, new_height)), scale

class ROISelector:
    def __init__(self):
        self.training_roi: List[Tuple[int, int, int, int]] = []
        self.reference_roi: List[Tuple[int, int, int, int]] = []
        
    def clear(self):
        self.training_roi = []
        self.reference_roi = []

def main():
    st.set_page_config(page_title="Image Analysis Tool", layout="wide")
    
    st.title("Image Analysis Tool")
    
    # Initialize session state
    if 'roi_selector' not in st.session_state:
        st.session_state.roi_selector = ROISelector()
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'file_names' not in st.session_state:
        st.session_state.file_names = []
    if 'scale_factors' not in st.session_state:
        st.session_state.scale_factors = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        uploaded_files = st.file_uploader(
            "Upload Images", 
            type=['png', 'jpg', 'jpeg', 'bmp'], 
            accept_multiple_files=True
        )
        
        if st.button("Clear All"):
            st.session_state.processed_images = []
            st.session_state.file_names = []
            st.session_state.scale_factors = []
            st.session_state.roi_selector.clear()
            st.experimental_rerun()
    
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        
        if current_files != st.session_state.file_names:
            st.session_state.processed_images = []
            st.session_state.file_names = []
            st.session_state.scale_factors = []
            
            for uploaded_file in uploaded_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.bmp') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                    
                    img = cv.imread(tmp_file.name)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    
                    os.unlink(tmp_file.name)
                    
                    if img is not None:
                        scaled_img, scale_factor = scale_image_for_display(img)
                        st.session_state.processed_images.append(img)
                        st.session_state.file_names.append(uploaded_file.name)
                        st.session_state.scale_factors.append(scale_factor)
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    if st.session_state.processed_images:
        st.write(f"Number of loaded images: {len(st.session_state.processed_images)}")
        
        selected_image_idx = st.selectbox(
            "Select Image",
            range(len(st.session_state.processed_images)),
            format_func=lambda x: f"{st.session_state.file_names[x]} (Image {x+1})"
        )
        
        current_image = st.session_state.processed_images[selected_image_idx]
        scale_factor = st.session_state.scale_factors[selected_image_idx]
        scaled_image, _ = scale_image_for_display(current_image)
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.write(f"Original dimensions: {current_image.shape}")
        with info_col2:
            st.write(f"Display dimensions: {scaled_image.shape}")
        with info_col3:
            st.write(f"Scale factor: {scale_factor:.3f}")
        
        # Training ROI
        st.subheader("Training ROI")
        pil_image = Image.fromarray(scaled_image)
        
        canvas_result_training = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=pil_image,
            drawing_mode="rect",
            update_streamlit=True,
            height=scaled_image.shape[0],
            width=scaled_image.shape[1],
            key="training_canvas",
        )
        
        if canvas_result_training.json_data is not None and len(canvas_result_training.json_data["objects"]) > 0:
            rect = canvas_result_training.json_data["objects"][-1]
            x1_train = int(rect["left"] / scale_factor)
            y1_train = int(rect["top"] / scale_factor)
            x2_train = int((rect["left"] + rect["width"]) / scale_factor)
            y2_train = int((rect["top"] + rect["height"]) / scale_factor)
            st.write(f"Training ROI (original coordinates): ({x1_train}, {y1_train}) to ({x2_train}, {y2_train})")
        
        # Reference ROI
        st.subheader("Reference ROI")
        canvas_result_reference = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=2,
            stroke_color="#00ff00",
            background_image=pil_image,
            drawing_mode="rect",
            update_streamlit=True,
            height=scaled_image.shape[0],
            width=scaled_image.shape[1],
            key="reference_canvas",
        )
        
        if canvas_result_reference.json_data is not None and len(canvas_result_reference.json_data["objects"]) > 0:
            rect = canvas_result_reference.json_data["objects"][-1]
            x1_ref = int(rect["left"] / scale_factor)
            y1_ref = int(rect["top"] / scale_factor)
            x2_ref = int((rect["left"] + rect["width"]) / scale_factor)
            y2_ref = int((rect["top"] + rect["height"]) / scale_factor)
            st.write(f"Reference ROI (original coordinates): ({x1_ref}, {y1_ref}) to ({x2_ref}, {y2_ref})")
        
        # Apply to all images button
        if st.button("Apply ROIs to all images"):
            if ('x1_train' in locals() and 'x1_ref' in locals()):
                st.session_state.roi_selector.training_roi = [
                    (x1_train, y1_train, x2_train, y2_train)
                ] * len(st.session_state.processed_images)
                st.session_state.roi_selector.reference_roi = [
                    (x1_ref, y1_ref, x2_ref, y2_ref)
                ] * len(st.session_state.processed_images)
                st.success("ROIs applied to all images")
            else:
                st.warning("Please draw both ROIs before applying to all images")

if __name__ == "__main__":
    main()