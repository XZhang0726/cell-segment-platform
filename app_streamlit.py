"""Cell Segmentation Platform: compact Streamlit interface.

Upload microscopy images and interactively configure cell segmentation."""
import sys
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import time

# Add the project root to the import path.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod


# Page configuration
st.set_page_config(
    page_title="Cell Segmentation Platform",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Cell Segmentation Platform")
st.markdown("Upload a cell microscopy image, select a segmentation method, and inspect the results.")

# Sidebar: inputs and parameters
with st.sidebar:
    st.header("⚙️ Settings")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Cell Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Supported formats: PNG, JPG, and TIFF."
    )

    # Segmentation method selection
    method = st.selectbox(
        "Segmentation Method",
        ["Otsu Threshold", "Adaptive Threshold", "Watershed Algorithm", "Canny Edge Detection"],
        help="Select a cell segmentation method."
    )

    # Advanced parameters
    with st.expander("🔧 Advanced Parameters"):
        if method == "Adaptive Threshold":
            block_size = st.slider(
                "Block Size (block_size)",
                min_value=3,
                max_value=51,
                value=11,
                step=2,
                help="Use an odd integer; larger values consider a wider neighborhood."
            )
            C = st.slider(
                "Constant (C)",
                min_value=0,
                max_value=20,
                value=2,
                help="Constant subtracted from the local mean."
            )
        elif method == "Canny Edge Detection":
            low_threshold = st.slider(
                "Low Threshold",
                min_value=0,
                max_value=200,
                value=50,
                step=10,
                help="Lower threshold for edge detection."
            )
            high_threshold = st.slider(
                "High Threshold",
                min_value=0,
                max_value=300,
                value=150,
                step=10,
                help="Upper threshold for edge detection."
            )

    # Segmentation button
    segment_button = st.button("🚀 Start Segmentation", type="primary", use_container_width=True)

# Main panel: results
if uploaded_file is not None:
    # Read the image.
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Display the original image.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image_np.shape}")

    # Run segmentation.
    if segment_button:
        with col2:
            st.subheader("📊 Segmentation Result")

            with st.spinner("Segmenting..."):
                try:
                    # Create the segmenter.
                    method_map = {
                        "Otsu Threshold": SegmentationMethod.OTSU,
                        "Adaptive Threshold": SegmentationMethod.ADAPTIVE,
                        "Watershed Algorithm": SegmentationMethod.WATERSHED,
                        "Canny Edge Detection": SegmentationMethod.EDGE_CANNY
                    }

                    seg_method = method_map[method]
                    segmenter = CellSegmenter(method=seg_method)

                    # Configure parameters.
                    params = {}
                    if seg_method == SegmentationMethod.ADAPTIVE:
                        params = {"block_size": int(block_size), "C": int(C)}
                    elif seg_method == SegmentationMethod.EDGE_CANNY:
                        params = {"low_threshold": int(low_threshold), "high_threshold": int(high_threshold)}

                    # Run segmentation.
                    start_time = time.time()
                    mask = segmenter.segment(image_np, **params)
                    elapsed_time = time.time() - start_time

                    # Normalize the mask.
                    if mask.max() > 0:
                        mask_display = (mask / mask.max() * 255).astype(np.uint8)
                    else:
                        mask_display = mask.astype(np.uint8)

                    # Create a color overlay.
                    if len(image_np.shape) == 2:
                        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                    else:
                        image_rgb = image_np.copy()

                    overlay = image_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

                    # Display results.
                    tab1, tab2 = st.tabs(["Segmentation Mask", "Overlay Display"])

                    with tab1:
                        st.image(mask_display, use_container_width=True)

                    with tab2:
                        st.image(result, use_container_width=True)

                    # Summary statistics
                    st.success("Segmentation complete.")

                    foreground_pixels = np.sum(mask > 0)
                    total_pixels = mask.size
                    foreground_ratio = foreground_pixels / total_pixels * 100

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Foreground Pixels", f"{foreground_pixels:,}")
                    with col_b:
                        st.metric("Foreground Ratio", f"{foreground_ratio:.2f}%")
                    with col_c:
                        st.metric("Processing Time", f"{elapsed_time*1000:.2f} ms")

                    if seg_method == SegmentationMethod.WATERSHED:
                        num_regions = len(np.unique(mask)) - 1
                        st.info(f"Detected {num_regions} cell regions")

                except Exception as e:
                    st.error(f"Segmentation failed: {str(e)}")
else:
    st.info("👈 Upload a cell image in the sidebar to begin.")

# User guide
with st.expander("📖 User Guide"):
    st.markdown("""
    ### Workflow
    1. Upload a cell microscopy image in the sidebar.
    2. Choose a segmentation method.
    3. Adjust advanced parameters (optional).
    4. Click "Start Segmentation".
    5. Review the segmentation results and statistics.

    ### Segmentation Methods
    - **Otsu Threshold**: Automatically estimates a global threshold for high-contrast images.
    - **Adaptive Threshold**: Uses local thresholds for images with uneven illumination.
    - **Watershed Algorithm**: Uses distance-based region segmentation to separate touching cells.
    - **Canny Edge Detection**: Detects edges in images with well-defined cell boundaries.
    """)
