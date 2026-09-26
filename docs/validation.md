# Validation record

This record describes checks performed on the English documentation and interface revision on 2026-09-26. It is a software smoke test, not a segmentation accuracy benchmark.

## Environment

- macOS on Apple silicon, CPU execution
- Python 3.12.14
- NumPy 1.26.4, OpenCV 4.11.0, scikit-image 0.26.0
- PyTorch 2.14.0, torchvision 0.29.0
- scikit-learn 1.9.1, Streamlit 1.64.0
- pytest 9.1.1

The checks used an isolated environment. This is a record of one environment, not a certified compatibility matrix for all dependency versions or platforms.

## Results

| Check | Result |
| --- | --- |
| Python syntax | All 78 Python source files parsed successfully |
| English text and filenames | No Chinese characters in the current source, documentation, configuration, or filenames |
| English interface catalog | All 883 static translation calls and 10 dynamic help sections resolved; formatting placeholders checked |
| Enhanced Streamlit application | Initialized with 10 English tabs and no application exception; all 7 method-selection branches checked |
| Minimal Streamlit application | Initialized without an exception; all 4 method-selection branches checked |
| Legacy language session | An old language preference falls back to English |
| Gradio interface | Constructed successfully; Otsu, adaptive thresholding, watershed, and Canny callbacks returned masks, overlays, and English statistics for a synthetic 96 × 96 image |
| Confidence-map demonstration | Both synthetic confidence-generation checks passed; example figures regenerated with English labels |
| Package build | Wheel built successfully; segmentation API and English catalog included |
| Automated suite | **79 passed, 3 failed**; the same three failures reproduced on the original revision |

The suite was run with:

```bash
python -m pytest tests -o addopts='' --tb=short -q
```

The original revision used for comparison was `0c796ca2a5470dd64a5e42a7dbed12ce12499919`. Source comparisons also verified that the English conversion did not change the segmentation, fusion, or machine learning algorithms.

## Known pre-existing test failures

| Test | Observed issue |
| --- | --- |
| `tests/test_config.py::TestTrainingConfig::test_save_config` | Configuration output contains a Python tuple tag that `yaml.safe_load` cannot read. |
| `tests/test_config.py::TestTrainingConfig::test_load_config` | The same tuple serialization issue prevents configuration round-tripping. |
| `tests/test_unet.py::TestUp::test_up_forward_bilinear` | The test supplies concatenated feature channels that do not match the convolution input channel count. |

These failures are not introduced by the English conversion and remain unresolved in this revision. Some machine learning tests catch exceptions and return booleans rather than asserting outcomes; their pytest pass status alone does not establish workflow correctness.

## Not covered

Pretrained Cellpose, CellViT, and SAM inference was not run, and no model weights were downloaded. CUDA execution, Windows launcher execution, real-dataset accuracy, training quality, uncertainty calibration, and end-to-end active learning performance were not evaluated. Interface method-selection checks do not establish model inference correctness.
