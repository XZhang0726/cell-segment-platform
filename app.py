"""Cell Segmentation Platform: Gradio interface.

Upload microscopy images and interactively configure cell segmentation."""
import sys
from pathlib import Path
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# Add the project root to the import path.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod


def segment_cell_image(
    image,
    method,
    block_size=11,
    C=2,
    low_threshold=50,
    high_threshold=150,
    model_path=None
):
    """Segment a microscopy image and prepare display outputs.

    Args:
        image: Input image as an array or PIL image.
        method: Display name of the segmentation method.
        block_size: Neighborhood size for adaptive thresholding.
        C: Constant subtracted from the local mean.
        low_threshold: Lower Canny hysteresis threshold.
        high_threshold: Upper Canny hysteresis threshold.
        model_path: Optional path to a deep learning model.

    Returns:
        Display mask, image overlay, and a Markdown statistics summary.
    """
    if image is None:
        return None, None, "Upload an image to begin."

    # Convert the image format.
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure the image uses uint8 values.
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    try:
        # Create the segmenter.
        method_map = {
            "Otsu Threshold": SegmentationMethod.OTSU,
            "Adaptive Threshold": SegmentationMethod.ADAPTIVE,
            "Watershed Algorithm": SegmentationMethod.WATERSHED,
            "Canny Edge Detection": SegmentationMethod.EDGE_CANNY,
            "Deep Learning": SegmentationMethod.DEEP_LEARNING
        }

        seg_method = method_map[method]
        segmenter = CellSegmenter(
            method=seg_method,
            model_path=model_path if seg_method == SegmentationMethod.DEEP_LEARNING else None
        )

        # Configure method-specific parameters.
        params = {}
        if seg_method == SegmentationMethod.ADAPTIVE:
            params = {"block_size": int(block_size), "C": int(C)}
        elif seg_method == SegmentationMethod.EDGE_CANNY:
            params = {"low_threshold": int(low_threshold), "high_threshold": int(high_threshold)}

        # Run segmentation.
        mask = segmenter.segment(image, **params)

        # Normalize the mask for display.
        if mask.max() > 0:
            mask_display = (mask / mask.max() * 255).astype(np.uint8)
        else:
            mask_display = mask.astype(np.uint8)

        # Create a color overlay.
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()

        # Create a red mask overlay.
        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

        # Summary statistics
        foreground_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        foreground_ratio = foreground_pixels / total_pixels * 100

        info = f"""
        ### Segmentation Statistics
        - **Method**: {method}
        - **Foreground Pixels**: {foreground_pixels:,}
        - **Total Pixels**: {total_pixels:,}
        - **Foreground Fraction**: {foreground_ratio:.2f}%
        """

        if seg_method == SegmentationMethod.WATERSHED:
            num_regions = len(np.unique(mask)) - 1
            info += f"\n- **Detected Regions**: {num_regions}"

        return mask_display, result, info

    except Exception as e:
        return None, None, f"Error: {str(e)}"


# Build the Gradio interface.
with gr.Blocks(title="Cell Segmentation Platform") as demo:
    gr.Markdown("""
    # 🔬 Cell Segmentation Platform

    Upload a cell microscopy image, select a segmentation method, and inspect the results.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Input panel
            gr.Markdown("### 📤 Input Image")
            input_image = gr.Image(
                label="Upload Cell Image",
                type="numpy",
                height=300
            )

            gr.Markdown("### ⚙️ Segmentation Settings")
            method = gr.Dropdown(
                choices=["Otsu Threshold", "Adaptive Threshold", "Watershed Algorithm", "Canny Edge Detection"],
                value="Otsu Threshold",
                label="Segmentation Method"
            )

            # Method-specific parameter controls
            with gr.Accordion("Advanced Parameters", open=False):
                block_size = gr.Slider(
                    minimum=3,
                    maximum=51,
                    step=2,
                    value=11,
                    label="Adaptive Threshold - Block Size",
                    info="Use an odd integer; larger values consider a wider neighborhood."
                )
                C = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=1,
                    value=2,
                    label="Adaptive Threshold - Constant (C)",
                    info="Constant subtracted from the local mean."
                )
                low_threshold = gr.Slider(
                    minimum=0,
                    maximum=200,
                    step=10,
                    value=50,
                    label="Canny - Lower Threshold",
                    info="Lower threshold for edge detection."
                )
                high_threshold = gr.Slider(
                    minimum=0,
                    maximum=300,
                    step=10,
                    value=150,
                    label="Canny - Upper Threshold",
                    info="Upper threshold for edge detection."
                )

            segment_btn = gr.Button("🚀 Start Segmentation", variant="primary", size="lg")

        with gr.Column(scale=2):
            # Output panel
            gr.Markdown("### 📊 Segmentation Results")

            with gr.Row():
                mask_output = gr.Image(label="Segmentation Mask", height=300)
                overlay_output = gr.Image(label="Overlay Display", height=300)

            info_output = gr.Markdown()

    # Example images
    gr.Markdown("### 💡 Examples")
    gr.Examples(
        examples=[
            ["Otsu Threshold", 11, 2, 50, 150],
            ["Adaptive Threshold", 15, 3, 50, 150],
            ["Watershed Algorithm", 11, 2, 50, 150],
            ["Canny Edge Detection", 11, 2, 30, 100],
        ],
        inputs=[method, block_size, C, low_threshold, high_threshold],
        label="Parameter Presets"
    )

    # User guide
    with gr.Accordion("📖 User Guide", open=False):
        gr.Markdown("""
        ### Workflow
        1. **Upload an image**: Select a cell microscopy image.
        2. **Choose a method**: Select a segmentation method from the dropdown.
        3. **Adjust parameters**: Expand "Advanced Parameters" to configure the selected method.
        4. **Run segmentation**: Click "Start Segmentation".
        5. **Inspect results**: Review the segmentation mask and image overlay.

        ### Segmentation Methods
        - **Otsu Threshold**: Automatically estimates a global threshold for high-contrast images.
        - **Adaptive Threshold**: Uses local thresholds for images with uneven illumination.
        - **Watershed Algorithm**: Uses distance-based region segmentation to separate touching cells.
        - **Canny Edge Detection**: Detects edges in images with well-defined cell boundaries.

        ### Suggested Parameter Ranges
        - **block_size**: Start with an odd integer from 11 to 35; larger images may benefit from a larger neighborhood.
        - **C**: Start with a value from 2 to 10 and inspect the resulting foreground mask.
        - **Canny thresholds**: Start with 20-50 for the lower threshold and 60-150 for the upper threshold.
        """)

    # Bind interface events.
    segment_btn.click(
        fn=segment_cell_image,
        inputs=[input_image, method, block_size, C, low_threshold, high_threshold],
        outputs=[mask_output, overlay_output, info_output]
    )


if __name__ == "__main__":
    # Launch with default settings.
    demo.launch(
        share=False,
        inbrowser=False
    )
