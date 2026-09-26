# CellViT Manual Test Protocol

Use this protocol to verify the repository's CellViT subprocess integration and inspect segmentation outputs. Functional success is separate from model accuracy on a scientific dataset.

## Preparation

Follow the [installation guide](installation.md#cellvit-worker-environment) and [CellViT guide](cellvit-guide.md). Confirm imports in a separate terminal with the worker environment active:

```bash
conda activate ./env_cellvit
python -c "import cellvit, numpy; print('CellViT import: OK'); print('NumPy:', numpy.__version__)"
conda deactivate
```

In the terminal where the main environment is active, launch the application:

```bash
python -m streamlit run app_enhanced.py
```

The main application does not need to run inside `env_cellvit`; the adapter selects that interpreter for the worker. Prepare a representative nuclei image, preferably at or below 256 x 256 pixels for the first check.

## 1. Single-image test

1. Upload the image and select CellViT.
2. Use `CellViT-256`, `target_size=256`, and CPU mode unless CUDA is available inside `env_cellvit`.
3. Run segmentation and inspect the captured worker output.
4. Confirm successful model loading, inference, and instance postprocessing.
5. Verify that the returned mask matches the input dimensions and aligns with the displayed image.
6. Inspect positive labels, nuclei boundaries, missed objects, fragments, and merged nuclei.

A fallback to the binary-map channel must be recorded as a postprocessing problem, even if an integer mask is returned. See [the implementation limits](cellvit-guide.md#model-loading-and-postprocessing).

## 2. Batch test

1. Upload two to five images in the batch-processing workflow.
2. Keep the model and preprocessing settings fixed.
3. Check progress, per-image errors, and filename-to-result correspondence.
4. Confirm that overlays, feature tables, and exported files match the inspected masks.

Each call starts another worker and reloads the model, so include process startup and model-loading time when reporting end-to-end batch throughput.

## 3. Image-size test

Use representative 128 x 128, 256 x 256, and 512 x 512 images with known object scale. Keep `target_size=256` for the first comparison.

Smaller images are padded; larger images are resized before inference. A restored output shape does not mean that original spatial detail was preserved. Compare counts and boundaries before considering other target sizes.

## 4. Device and timing test

On a compatible NVIDIA machine, compare CPU and CUDA with identical inputs and settings. Verify CUDA inside the worker environment rather than only in the main environment.

Record first-run downloads separately. Time model loading, inference, postprocessing, and full end-to-end processing where possible. Define workload-specific acceptance criteria rather than assuming a fixed GPU speedup.

| Image / dimensions | Target size | Device | Model download needed | Full time | Positive labels | Postprocessing status / quality |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Troubleshooting

- **Missing worker environment:** verify the repository-local `env_cellvit` directory and Python executable.
- **Import failure:** activate the worker environment, run the import check, and inspect `python -m pip check`.
- **Download failure:** inspect network access, cache path, and disk space.
- **CUDA memory error:** select CPU or reduce concurrent GPU workloads.
- **Timeout:** the wrapper allows five minutes; use worker logs to identify the slow phase.
- **Zero or implausible detections:** inspect input quality, nuclei scale, RGB intensity range, modality, and fallback warnings.

## Checklist

- [ ] The worker environment and Python executable exist.
- [ ] CellViT imports in the worker environment.
- [ ] The main application starts in its own environment.
- [ ] Model loading, inference, and instance postprocessing succeed.
- [ ] Single-image and batch outputs are correctly associated with inputs.
- [ ] Output dimensions and visualization are correct.
- [ ] Actual compute device is recorded.
- [ ] Image scaling effects are inspected.
- [ ] Feature extraction and exports are checked.
- [ ] Accuracy is evaluated against annotations where available.

## Reporting an issue

Include the repository revision, operating system, Python/NumPy/CellViT/PyTorch versions in the worker environment, the selected parameters, and the full worker stdout/stderr. Provide a shareable image and reproduction steps when possible through the [project issue tracker](https://github.com/XZhang0726/cell-segment-platform/issues).
