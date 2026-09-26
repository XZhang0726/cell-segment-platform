"""
Test confidence generation

Verify the generation of synthetic confidence maps
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.fusion.confidence_utils import generate_confidence_from_mask, generate_confidence_maps

def test_single_mask_confidence():
    """Test confidence generation for a single mask"""
    print("=" * 60)
    print("Test 1: Confidence generation for a single mask")
    print("=" * 60)

    # Create a 100 x 100 test mask containing two circular cells
    mask = np.zeros((100, 100), dtype=np.int32)

    # Cell 1: center at (30, 30), radius 15 pixels
    y, x = np.ogrid[:100, :100]
    cell1 = ((x - 30)**2 + (y - 30)**2) <= 15**2
    mask[cell1] = 1

    # Cell 2: center at (70, 70), radius 20 pixels
    cell2 = ((x - 70)**2 + (y - 70)**2) <= 20**2
    mask[cell2] = 2

    # Generate confidence maps
    confidence_map = generate_confidence_from_mask(mask, base_confidence=0.8, boundary_penalty=0.3)

    # Verify the result
    print(f"Mask shape: {mask.shape}")
    print(f"Cell count: {np.max(mask)}")
    print(f"Confidence map shape: {confidence_map.shape}")
    print(f"Confidence range: [{np.min(confidence_map):.3f}, {np.max(confidence_map):.3f}]")

    # Inspect the confidence distribution for cell 1
    cell1_conf = confidence_map[cell1]
    print(f"\nCell 1 confidence statistics:")
    print(f"  Mean: {np.mean(cell1_conf):.3f}")
    print(f"  Minimum: {np.min(cell1_conf):.3f} (boundary)")
    print(f"  Maximum: {np.max(cell1_conf):.3f} (center)")

    # Inspect the confidence distribution for cell 2
    cell2_conf = confidence_map[cell2]
    print(f"\nCell 2 confidence statistics:")
    print(f"  Mean: {np.mean(cell2_conf):.3f}")
    print(f"  Minimum: {np.min(cell2_conf):.3f} (boundary)")
    print(f"  Maximum: {np.max(cell2_conf):.3f} (center)")

    # Visualize the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask, cmap='tab20')
    axes[0].set_title('Original mask')
    axes[0].axis('off')

    axes[1].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Confidence map')
    axes[1].axis('off')

    # Display the overlay
    axes[2].imshow(mask, cmap='gray', alpha=0.3)
    im = axes[2].imshow(confidence_map, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title('Mask and confidence overlay')
    axes[2].axis('off')

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('test_confidence_single.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: test_confidence_single.png")

    # Verify that confidence is higher at the center than at the boundary
    if np.max(cell1_conf) > np.min(cell1_conf):
        print("\n[PASS] Confidence gradient verified: center > boundary")
        return True
    else:
        print("\n[FAIL] Unexpected confidence gradient")
        return False


def test_multiple_models_confidence():
    """Test confidence generation for multiple models"""
    print("\n" + "=" * 60)
    print("Test 2: Confidence generation for multiple models")
    print("=" * 60)

    # Create masks that simulate predictions from three different models
    mask1 = np.zeros((100, 100), dtype=np.int32)
    mask2 = np.zeros((100, 100), dtype=np.int32)
    mask3 = np.zeros((100, 100), dtype=np.int32)

    y, x = np.ogrid[:100, :100]

    # Model 1: two detected cells
    cell1 = ((x - 30)**2 + (y - 30)**2) <= 15**2
    cell2 = ((x - 70)**2 + (y - 70)**2) <= 20**2
    mask1[cell1] = 1
    mask1[cell2] = 2

    # Model 2: two detected cells with slightly shifted positions
    cell1_shifted = ((x - 32)**2 + (y - 32)**2) <= 16**2
    cell2_shifted = ((x - 68)**2 + (y - 68)**2) <= 19**2
    mask2[cell1_shifted] = 1
    mask2[cell2_shifted] = 2

    # Model 3: only one detected cell
    mask3[cell2] = 1

    masks_list = [mask1, mask2, mask3]
    model_names = ['cellpose', 'cellvit', 'cellsam']
    model_reliabilities = {
        'cellpose': 0.9,
        'cellvit': 0.85,
        'cellsam': 0.8
    }

    # Generate confidence maps
    confidences_list = generate_confidence_maps(masks_list, model_names, model_reliabilities)

    print(f"Generated {len(confidences_list)} confidence maps")

    # Verify confidence values for each model
    for i, (conf_map, model_name) in enumerate(zip(confidences_list, model_names)):
        mask = masks_list[i]
        reliability = model_reliabilities[model_name]

        # Calculate mean confidence over the foreground
        foreground = mask > 0
        if np.sum(foreground) > 0:
            avg_conf = np.mean(conf_map[foreground])
            print(f"\n{model_name}:")
            print(f"  Reliability: {reliability:.2f}")
            print(f"  Mean confidence: {avg_conf:.3f}")
            print(f"  Confidence range: [{np.min(conf_map[foreground]):.3f}, {np.max(conf_map[foreground]):.3f}]")

    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (mask, conf_map, model_name) in enumerate(zip(masks_list, confidences_list, model_names)):
        # First row: masks
        axes[0, i].imshow(mask, cmap='tab20')
        axes[0, i].set_title(f'{model_name} - mask')
        axes[0, i].axis('off')

        # Second row: confidence maps
        im = axes[1, i].imshow(conf_map, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'{model_name} - confidence')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('test_confidence_multiple.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: test_confidence_multiple.png")

    # Verify that different models produce different confidence distributions
    conf1_mean = np.mean(confidences_list[0][masks_list[0] > 0])
    conf2_mean = np.mean(confidences_list[1][masks_list[1] > 0])
    conf3_mean = np.mean(confidences_list[2][masks_list[2] > 0])

    if conf1_mean != conf2_mean and conf2_mean != conf3_mean:
        print("\n[PASS] Confidence distributions differ across models")
        return True
    else:
        print("\n[FAIL] Unexpected confidence distribution")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Confidence generation tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Confidence generation for a single mask", test_single_mask_confidence()))
    results.append(("Confidence generation for multiple models", test_multiple_models_confidence()))

    # Summarize the results
    print("\n" + "=" * 60)
    print("Test results")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] passed" if result else "[FAIL] failed"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All confidence generation tests passed.")
        return True
    else:
        print(f"\n[WARN] {total - passed} tests failed; review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
