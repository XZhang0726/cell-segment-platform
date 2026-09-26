"""
U-Net model unit tests
"""
import pytest
import torch
import sys
from pathlib import Path

# Add the project root to the import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models.unet import UNet, DoubleConv, Down, Up, OutConv


class TestDoubleConv:
    """Test the DoubleConv module"""

    def test_double_conv_forward(self):
        """Test the DoubleConv forward pass"""
        module = DoubleConv(3, 64)
        x = torch.randn(2, 3, 256, 256)
        output = module(x)

        assert output.shape == (2, 64, 256, 256)
        assert output.dtype == torch.float32

    def test_double_conv_with_mid_channels(self):
        """Test intermediate channels in DoubleConv"""
        module = DoubleConv(3, 64, mid_channels=32)
        x = torch.randn(2, 3, 256, 256)
        output = module(x)

        assert output.shape == (2, 64, 256, 256)


class TestDown:
    """Test the Down module"""

    def test_down_forward(self):
        """Test the Down forward pass"""
        module = Down(64, 128)
        x = torch.randn(2, 64, 256, 256)
        output = module(x)

        # Downsampling halves the spatial dimensions
        assert output.shape == (2, 128, 128, 128)


class TestUp:
    """Test the Up module"""

    def test_up_forward_bilinear(self):
        """Test bilinear upsampling"""
        module = Up(128, 64, bilinear=True)
        x1 = torch.randn(2, 128, 64, 64)
        x2 = torch.randn(2, 64, 128, 128)
        output = module(x1, x2)

        assert output.shape == (2, 64, 128, 128)

    def test_up_forward_transpose(self):
        """Test transposed-convolution upsampling"""
        module = Up(128, 64, bilinear=False)
        x1 = torch.randn(2, 128, 64, 64)
        x2 = torch.randn(2, 64, 128, 128)
        output = module(x1, x2)

        assert output.shape == (2, 64, 128, 128)


class TestOutConv:
    """Test the OutConv module"""

    def test_out_conv_forward(self):
        """Test the OutConv forward pass"""
        module = OutConv(64, 1)
        x = torch.randn(2, 64, 256, 256)
        output = module(x)

        assert output.shape == (2, 1, 256, 256)


class TestUNet:
    """Test the complete U-Net model"""

    def test_unet_forward_default(self):
        """Test the U-Net forward pass with the default configuration"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)
        assert output.dtype == torch.float32

    def test_unet_forward_grayscale(self):
        """Test grayscale input to U-Net"""
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        x = torch.randn(2, 1, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_forward_multiclass(self):
        """Test multiclass output from U-Net"""
        model = UNet(n_channels=3, n_classes=5, bilinear=True)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 5, 256, 256)

    def test_unet_forward_transpose_conv(self):
        """Test transposed convolutions in U-Net"""
        model = UNet(n_channels=3, n_classes=1, bilinear=False)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_different_input_sizes(self):
        """Test different input sizes"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)

        # Test 512x512
        x = torch.randn(1, 3, 512, 512)
        output = model(x)
        assert output.shape == (1, 1, 512, 512)

        # Test 128x128
        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        assert output.shape == (1, 1, 128, 128)

    def test_unet_batch_sizes(self):
        """Test different batch sizes"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)

        # Batch size: 1
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape == (1, 1, 256, 256)

        # Batch size: 8
        x = torch.randn(8, 3, 256, 256)
        output = model(x)
        assert output.shape == (8, 1, 256, 256)

    def test_unet_gradient_flow(self):
        """Test gradient flow"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that the input receives gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
