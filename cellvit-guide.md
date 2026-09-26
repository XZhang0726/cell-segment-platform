# CellViT Integration Guide

CellViT is a transformer-based nuclei-segmentation model intended primarily for histopathology images. This repository provides an adapter around CellViT-256 and returns a label map through the common segmentation API.

## Separate worker environment

The adapter starts a subprocess using `env_cellvit` in the repository root. The main Streamlit application can stay in its normal environment. This isolation avoids mixing CellViT dependencies with the main application's packages.

Cloning the repository does not create this environment. Follow the [CellViT installation steps](installation.md#cellvit-worker-environment), then verify it in a separate terminal:

```bash
conda activate ./env_cellvit
python -c "import cellvit, numpy; print('CellViT import: OK'); print('NumPy:', numpy.__version__)"
conda deactivate
```

The adapter targets `cellvit==1.0.9`, with `numpy==1.26.4` as the worker dependency baseline. Use a compatible PyTorch build and verify inference in your environment; the repository does not provide a complete cross-platform compatibility matrix.

## Python API

Run from the repository root and replace the example image path:

```python
import numpy as np
from PIL import Image
from src.api.segmentation import CellSegmenter, SegmentationMethod

image = np.asarray(Image.open("path/to/image.png").convert("RGB"))
segmenter = CellSegmenter(method=SegmentationMethod.CELLVIT)
masks = segmenter.segment(
    image,
    model_type="CellViT-256",
    use_gpu=False,
    target_size=256,
)
```

### Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_type` | `"CellViT-256"` | Public model selector; the current worker instantiates CellViT256 |
| `use_gpu` | `False` | Request CUDA if it is available inside the worker environment |
| `target_size` | `256` | Padded/resized input size; use 256 for the documented CellViT-256 workflow |
| `progress_bar` | `None` | Optional Streamlit progress object |

Changing `model_type` alone does not add support for another CellViT architecture. The current worker does not select Apple MPS.

## Streamlit workflow

From the terminal where your main application environment is active:

```bash
python -m streamlit run app_enhanced.py
```

Upload an image and select CellViT. Begin with CellViT-256 and a target size of 256. Check the worker output for the actual device, model loading, inference, and postprocessing status.

## Image adaptation

The adapter is intended for ordinary image inputs; it is not a whole-slide processing pipeline.

1. The wrapper expands grayscale inputs to RGB and removes alpha from RGBA inputs.
2. The worker downsizes images larger than `target_size` while preserving aspect ratio.
3. It pads the resulting image to the target size.
4. It divides intensities by 255 before inference, so provide RGB data on the expected 8-bit intensity scale.
5. It removes padding and resizes the mask back to the input dimensions with nearest-neighbor interpolation.

Downscaling a large image to a single 256-pixel input can remove small nuclei and morphological detail. Validate the chosen image scale before interpreting cell counts or measurements.

## Model loading and postprocessing

The worker calls CellViT's `cache_cellvit_256()` helper. A first run may need network access and a model download. Each wrapper call launches a new worker, which loads the model again; there is no persistent in-process model cache shared across these worker calls.

The worker attempts CellViT instance postprocessing using nuclei, type, and horizontal/vertical maps. It constructs minimal slide metadata with `pixel_size=1.0`; this is not a substitute for validated acquisition metadata.

If instance postprocessing fails, the current implementation logs a warning and falls back to a binary-map channel before returning an integer array. **Inspect those warnings:** a returned array alone does not prove that valid instance segmentation succeeded. Do not use a fallback result as a verified cell count.

The subprocess has a five-minute timeout. Large downloads or slow CPU execution may exceed it.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `CellViT environment not found` | Create `env_cellvit` at the repository root |
| Python executable missing | Check `env_cellvit/bin/python` on macOS/Linux, or the Windows paths in the installation guide |
| CellViT import failure | Test imports inside the worker environment, then inspect dependency versions |
| CUDA out of memory | Try CPU mode and smaller inputs; reduce competing GPU workloads |
| Model download failure | Check network access, model-cache location, and available disk space |
| Timeout | Read captured worker stdout/stderr to distinguish downloading, loading, and inference |
| Empty or implausible mask | Check input modality, image scale, dtype, and whether instance postprocessing fell back |

See the [manual test protocol](cellvit-testing.md) for an end-to-end checklist.

## Choosing a backend

CellViT focuses on nuclei in histopathology images. Cellpose provides models for general cell and nuclei segmentation. The repository's SAM wrapper performs general-purpose automatic object segmentation. Compare them on your own annotated data; the repository does not establish a universal ranking of their speed or accuracy.

## References

- [CellViT source](https://github.com/TIO-IKIM/CellViT)
- [CellViT inference documentation](https://tio-ikim.github.io/CellViT-Inference/)
- [CellViT package](https://pypi.org/project/cellvit/)
