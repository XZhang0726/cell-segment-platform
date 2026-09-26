"""
Unit tests for training configuration
"""
import pytest
import yaml
import sys
from pathlib import Path
import tempfile

# Add the project root to the import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.config import TrainingConfig, get_default_config


class TestTrainingConfig:
    """Test the training configuration"""

    def test_default_config(self):
        """Test the default configuration"""
        config = TrainingConfig()

        # Check the default values
        assert config.model_name == "unet"
        assert config.n_channels == 3
        assert config.n_classes == 1
        assert config.epochs == 100
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4

    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            model_name="custom_unet",
            epochs=50,
            batch_size=16,
            learning_rate=1e-3
        )

        assert config.model_name == "custom_unet"
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3

    def test_save_config(self):
        """Test saving the configuration"""
        config = TrainingConfig(epochs=50, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config.save(str(config_path))

            # Check that the file exists
            assert config_path.exists()

            # Check the file contents
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_data = yaml.safe_load(f)

            assert saved_data['epochs'] == 50
            assert saved_data['batch_size'] == 16

    def test_load_config(self):
        """Test loading the configuration"""
        config = TrainingConfig(epochs=50, batch_size=16, learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config.save(str(config_path))

            # Load the configuration
            loaded_config = TrainingConfig.load(str(config_path))

            # Check the loaded configuration
            assert loaded_config.epochs == 50
            assert loaded_config.batch_size == 16
            assert loaded_config.learning_rate == 1e-3

    def test_load_nonexistent_config(self):
        """Test loading a missing configuration file"""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load("nonexistent_config.yaml")


class TestGetDefaultConfig:
    """Test the default configuration factory"""

    def test_get_default_config(self):
        """Test default configuration retrieval"""
        config = get_default_config()

        # The result should be a TrainingConfig instance
        assert isinstance(config, TrainingConfig)

        # Check the default values
        assert config.model_name == "unet"
        assert config.epochs == 100
        assert config.batch_size == 8
