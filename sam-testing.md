# SAM Manual Test Protocol

This protocol tests the repository's SAM automatic-mask wrapper, exposed under the legacy `CELLSAM` API identifier. It does not validate the separate cell-trained CellSAM model or establish scientific accuracy by itself.

## Preparation

1. Follow the [installation guide](installation.md) and [SAM guide](sam-guide.md).
2. Activate the main application environment and verify the imports.
3. Put the selected SAM checkpoint in `models/sam/`.
4. Prepare a small representative image, preferably at or below 512 x 512 pixels, and a few additional images for batch testing. Retain their original pixel values and annotations for comparisons.

```bash
python -c "import segment_anything, torch; print('SAM dependencies: OK')"
python -m streamlit run app_enhanced.py
```

## 1. Single-image segmentation

1. Upload an image and select the SAM method.
2. Start with `vit_b`, point density 32, and CPU mode unless CUDA is available.
3. Run segmentation and confirm that the checkpoint loads without an exception.
4. Check that the mask dimensions match the input and that the overlay is aligned.
5. Count positive unique labels and compare their boundaries with the source image.
6. Record missed objects, false positives, fragments, merged objects, and unusual large-object filtering.

A nonempty result is a functional check, not evidence of accurate cell segmentation. If annotated masks are available, also measure the relevant pixel-level and instance-level metrics.

## 2. Batch processing

1. Open the batch-processing workflow.
2. Upload two to five images and choose the same backend settings.
3. Run the batch and inspect per-image progress and any failures.
4. Check that results remain associated with the correct filenames.
5. Verify visualization, feature extraction, and exported files for at least one image.

## 3. Checkpoint comparison

If the additional weights are available, repeat the same images with `vit_l` and `vit_h`. Keep preprocessing, point density, hardware, and output checks unchanged. Record the checkpoint, elapsed time, peak memory if available, and segmentation metrics.

Do not assume that larger models yield better masks. Checkpoint parameter counts are not file sizes, and neither is a measured accuracy score.

## 4. Parameter sensitivity

Compare `points_per_side=16`, `32`, and `48` on identical inputs. Optionally test 24 or 64 to explore a useful range. For each setting, record:

- Object count and agreement with annotations.
- Missed cells, fragments, and merged cells.
- Elapsed time and device.
- The effect of any denoising or contrast adjustment.

The current wrapper also applies fixed quality and area filters described in [sam-guide.md](sam-guide.md#implementation-details-and-limits). A change in point density cannot override those filters.

## 5. Device and caching checks

On a compatible NVIDIA machine, compare CPU and CUDA using the same model and image. The application should report the actual selected device. On macOS, a CPU result is expected for this wrapper.

Measure the first call separately from subsequent calls. The model is cached within the current Python process, so warm runs omit model-loading work. Restarting the application clears that in-memory cache.

## Troubleshooting

- **Checkpoint missing:** check the exact model filename and repository-relative `models/sam/` directory.
- **GPU memory exhausted:** use a smaller checkpoint, fewer sampling points, or CPU mode.
- **Too few detections:** review contrast, scale, sampling density, and the large-mask filter.
- **Too many fragments:** inspect image noise, preprocessing, sampling density, and overlap assignment.
- **Model loads but results are poor:** assess whether general-purpose SAM masks suit the target imaging modality; a successful model call is not sufficient validation.

## Test record

Record measurements from your own workload. Define acceptable latency and accuracy for the target images before interpreting the results.

| Image / dimensions | Checkpoint | Point density | Device | Cold / warm | Time | Positive labels | Quality notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

Checklist:

- [ ] Imports and checkpoint loading succeed.
- [ ] The application launches.
- [ ] Single-image and batch workflows complete.
- [ ] Output shape and overlay alignment are correct.
- [ ] The selected device is confirmed.
- [ ] Checkpoint and parameter comparisons are recorded.
- [ ] Feature extraction and exports are inspected.
- [ ] Accuracy is assessed separately from functional execution.

## Reporting an issue

Include the repository revision, operating system, Python/NumPy/PyTorch versions, selected checkpoint, parameters, actual device, and complete error text. Include a shareable example image and exact reproduction steps when possible. Report issues in the [project issue tracker](https://github.com/XZhang0726/cell-segment-platform/issues).
