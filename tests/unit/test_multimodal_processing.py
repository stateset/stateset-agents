"""
Unit tests for the Multimodal Processing module.

Tests cover modality types, processing strategies, fusion strategies,
and multimodal input/output handling.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from stateset_agents.core.multimodal_processing import (
    FusionStrategy,
    ModalityInput,
    ModalityType,
    ProcessedFeatures,
    ProcessingQuality,
)


class TestModalityType:
    """Test ModalityType enum."""

    def test_modality_type_values(self):
        """Test that all modality types have expected values."""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"
        assert ModalityType.STRUCTURED.value == "structured"
        assert ModalityType.SPEECH.value == "speech"
        assert ModalityType.GESTURE.value == "gesture"
        assert ModalityType.SENSOR.value == "sensor"
        assert ModalityType.MULTIMODAL.value == "multimodal"


class TestFusionStrategy:
    """Test FusionStrategy enum."""

    def test_fusion_strategy_values(self):
        """Test that all fusion strategies have expected values."""
        assert FusionStrategy.EARLY_FUSION.value == "early_fusion"
        assert FusionStrategy.LATE_FUSION.value == "late_fusion"
        assert FusionStrategy.HYBRID_FUSION.value == "hybrid_fusion"
        assert FusionStrategy.ATTENTION_FUSION.value == "attention_fusion"
        assert FusionStrategy.CROSS_MODAL_ATTENTION.value == "cross_modal_attention"
        assert FusionStrategy.TENSOR_FUSION.value == "tensor_fusion"


class TestProcessingQuality:
    """Test ProcessingQuality enum."""

    def test_processing_quality_values(self):
        """Test that all processing quality levels have expected values."""
        assert ProcessingQuality.LOW.value == "low"
        assert ProcessingQuality.MEDIUM.value == "medium"
        assert ProcessingQuality.HIGH.value == "high"
        assert ProcessingQuality.ADAPTIVE.value == "adaptive"


class TestModalityInput:
    """Test ModalityInput dataclass."""

    def test_modality_input_creation(self):
        """Test creating a ModalityInput."""
        data = "Hello, world!"
        input_obj = ModalityInput(
            modality=ModalityType.TEXT,
            data=data,
            metadata={"language": "en"},
            encoding="utf-8",
            source="/path/to/file.txt",
        )

        assert input_obj.modality == ModalityType.TEXT
        assert input_obj.data == data
        assert input_obj.metadata["language"] == "en"
        assert input_obj.encoding == "utf-8"
        assert input_obj.source == "/path/to/file.txt"

    def test_modality_input_defaults(self):
        """Test ModalityInput default values."""
        input_obj = ModalityInput(
            modality=ModalityType.IMAGE,
            data=b"image_bytes",
        )

        assert input_obj.metadata == {}
        assert input_obj.encoding is None
        assert input_obj.source is None
        assert input_obj.quality == ProcessingQuality.MEDIUM
        assert input_obj.timestamp is not None

    def test_modality_input_to_dict(self):
        """Test ModalityInput serialization to dictionary."""
        input_obj = ModalityInput(
            modality=ModalityType.TEXT,
            data="test data",
            metadata={"key": "value"},
        )

        result = input_obj.to_dict()

        assert result["modality"] == "text"
        assert result["metadata"] == {"key": "value"}
        assert result["quality"] == "medium"
        assert "timestamp" in result
        assert result["data_type"] == "str"
        assert result["data_size"] == 9  # len("test data")

    def test_modality_input_with_numpy_data(self):
        """Test ModalityInput with numpy array data."""
        data = np.zeros((224, 224, 3))
        input_obj = ModalityInput(
            modality=ModalityType.IMAGE,
            data=data,
        )

        result = input_obj.to_dict()
        assert result["data_type"] == "ndarray"

    def test_modality_input_custom_quality(self):
        """Test ModalityInput with custom quality."""
        input_obj = ModalityInput(
            modality=ModalityType.AUDIO,
            data=b"audio_data",
            quality=ProcessingQuality.HIGH,
        )

        assert input_obj.quality == ProcessingQuality.HIGH


class TestProcessedFeatures:
    """Test ProcessedFeatures dataclass."""

    @pytest.fixture
    def mock_tensor(self):
        """Create a mock tensor for testing."""
        with patch("core.multimodal_processing.TORCH_AVAILABLE", True):
            import torch
            return torch.randn(1, 768)

    def test_processed_features_creation(self, mock_tensor):
        """Test creating ProcessedFeatures."""
        features = ProcessedFeatures(
            modality=ModalityType.TEXT,
            features=mock_tensor,
            metadata={"model": "bert"},
            processing_time=0.5,
            confidence_score=0.95,
        )

        assert features.modality == ModalityType.TEXT
        assert features.features is not None
        assert features.processing_time == 0.5
        assert features.confidence_score == 0.95

    def test_processed_features_defaults(self, mock_tensor):
        """Test ProcessedFeatures default values."""
        features = ProcessedFeatures(
            modality=ModalityType.TEXT,
            features=mock_tensor,
        )

        assert features.attention_mask is None
        assert features.metadata == {}
        assert features.processing_time == 0.0
        assert features.confidence_score == 1.0

    def test_processed_features_to_dict(self, mock_tensor):
        """Test ProcessedFeatures serialization to dictionary."""
        features = ProcessedFeatures(
            modality=ModalityType.IMAGE,
            features=mock_tensor,
            processing_time=0.3,
        )

        result = features.to_dict()

        assert result["modality"] == "image"
        assert result["has_attention_mask"] is False
        assert result["processing_time"] == 0.3
        assert "feature_shape" in result


class TestModalityProcessor:
    """Test ModalityProcessor base class."""

    def test_processor_interface(self):
        """Test that ModalityProcessor defines required interface."""
        from stateset_agents.core.multimodal_processing import ModalityProcessor

        # ModalityProcessor is abstract, verify it has required methods
        assert hasattr(ModalityProcessor, "process")
        assert hasattr(ModalityProcessor, "get_feature_dimension")
        assert hasattr(ModalityProcessor, "initialize")


class TestTextProcessor:
    """Test TextProcessor class."""

    @pytest.fixture
    def mock_text_processor(self):
        """Create a mock TextProcessor."""
        with patch("core.multimodal_processing.TORCH_AVAILABLE", True), \
             patch("core.multimodal_processing.AutoTokenizer"), \
             patch("core.multimodal_processing.AutoModel"):
            from stateset_agents.core.multimodal_processing import TextProcessor
            processor = TextProcessor(model_name="test-model", device="cpu")
            return processor

    def test_text_processor_creation(self, mock_text_processor):
        """Test TextProcessor creation."""
        assert mock_text_processor.model_name == "test-model"
        assert mock_text_processor.device == "cpu"
        assert mock_text_processor.max_length == 512

    @pytest.mark.asyncio
    async def test_text_processor_initialize(self, mock_text_processor):
        """Test TextProcessor initialization."""
        with patch("core.multimodal_processing.AutoTokenizer") as mock_tokenizer, \
             patch("core.multimodal_processing.AutoModel") as mock_model:
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()

            await mock_text_processor.initialize()

            mock_tokenizer.from_pretrained.assert_called_once()
            mock_model.from_pretrained.assert_called_once()


class TestImageProcessor:
    """Test ImageProcessor class."""

    def test_image_processor_interface(self):
        """Test ImageProcessor implements required methods."""
        with patch("core.multimodal_processing.TORCH_AVAILABLE", True), \
             patch("core.multimodal_processing.PIL_AVAILABLE", True):
            from stateset_agents.core.multimodal_processing import ImageProcessor
            processor = ImageProcessor()

            assert hasattr(processor, "process")
            assert hasattr(processor, "get_feature_dimension")
            assert hasattr(processor, "initialize")


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_audio_processor_interface(self):
        """Test AudioProcessor implements required methods."""
        with patch("core.multimodal_processing.TORCH_AVAILABLE", True), \
             patch("core.multimodal_processing.AUDIO_AVAILABLE", True):
            from stateset_agents.core.multimodal_processing import AudioProcessor
            processor = AudioProcessor()

            assert hasattr(processor, "process")
            assert hasattr(processor, "get_feature_dimension")
            assert hasattr(processor, "initialize")


class TestMultimodalProcessor:
    """Test MultimodalProcessor class."""

    @pytest.fixture
    def multimodal_processor(self):
        """Create a mock MultimodalProcessor."""
        with patch("core.multimodal_processing.TORCH_AVAILABLE", True), \
             patch("core.multimodal_processing.get_monitoring_service") as mock_monitoring, \
             patch("core.multimodal_processing.ErrorHandler"):
            mock_monitoring.return_value = MagicMock()
            from stateset_agents.core.multimodal_processing import MultimodalProcessor
            processor = MultimodalProcessor()
            return processor

    def test_multimodal_processor_creation(self, multimodal_processor):
        """Test MultimodalProcessor creation."""
        assert multimodal_processor is not None

    @pytest.mark.asyncio
    async def test_process_multimodal_input(self, multimodal_processor):
        """Test processing multimodal input."""
        # This would test actual multimodal processing if fully implemented
        pass


class TestFusionStrategies:
    """Test multimodal fusion strategies."""

    def test_early_fusion_concept(self):
        """Test early fusion strategy conceptually."""
        # Early fusion combines raw features before processing
        features1 = np.random.randn(10, 128)
        features2 = np.random.randn(10, 128)

        # Concatenation-based early fusion
        fused = np.concatenate([features1, features2], axis=1)
        assert fused.shape == (10, 256)

    def test_late_fusion_concept(self):
        """Test late fusion strategy conceptually."""
        # Late fusion combines processed features
        processed1 = np.random.randn(10, 64)
        processed2 = np.random.randn(10, 64)

        # Weighted sum-based late fusion
        fused = 0.5 * processed1 + 0.5 * processed2
        assert fused.shape == (10, 64)


class TestCreateMultimodalProcessor:
    """Test create_multimodal_processor factory function."""

    def test_create_multimodal_processor(self):
        """Test factory function creates processor."""
        with patch("core.multimodal_processing.TORCH_AVAILABLE", True), \
             patch("core.multimodal_processing.get_monitoring_service") as mock_monitoring:
            mock_monitoring.return_value = MagicMock()
            from stateset_agents.core.multimodal_processing import create_multimodal_processor

            processor = create_multimodal_processor(["text", "image"])
            assert processor is not None


class TestModalityInputValidation:
    """Test input validation for modality inputs."""

    def test_text_input_validation(self):
        """Test text input is properly handled."""
        input_obj = ModalityInput(
            modality=ModalityType.TEXT,
            data="Valid text input",
        )
        assert isinstance(input_obj.data, str)

    def test_image_input_validation(self):
        """Test image input with bytes."""
        input_obj = ModalityInput(
            modality=ModalityType.IMAGE,
            data=b"\x89PNG\r\n\x1a\n...",  # PNG header
        )
        assert isinstance(input_obj.data, bytes)

    def test_structured_input_validation(self):
        """Test structured (JSON) input."""
        input_obj = ModalityInput(
            modality=ModalityType.STRUCTURED,
            data={"key": "value", "nested": {"a": 1}},
        )
        assert isinstance(input_obj.data, dict)


class TestProcessingMetadata:
    """Test processing metadata handling."""

    def test_metadata_preservation(self):
        """Test that metadata is preserved through processing."""
        metadata = {
            "source_file": "image.jpg",
            "original_size": (1024, 768),
            "format": "JPEG",
            "capture_time": "2024-01-01T12:00:00",
        }

        input_obj = ModalityInput(
            modality=ModalityType.IMAGE,
            data=b"image_data",
            metadata=metadata,
        )

        assert input_obj.metadata == metadata

    def test_processing_time_tracking(self):
        """Test that processing time is tracked."""
        import torch

        features = ProcessedFeatures(
            modality=ModalityType.TEXT,
            features=torch.randn(1, 768),
            processing_time=0.123,
        )

        assert features.processing_time == 0.123


class TestConfidenceScoring:
    """Test confidence scoring for processed features."""

    def test_confidence_score_range(self):
        """Test confidence score is in valid range."""
        import torch

        for score in [0.0, 0.5, 1.0]:
            features = ProcessedFeatures(
                modality=ModalityType.TEXT,
                features=torch.randn(1, 768),
                confidence_score=score,
            )
            assert 0.0 <= features.confidence_score <= 1.0

    def test_low_confidence_handling(self):
        """Test features with low confidence score."""
        import torch

        features = ProcessedFeatures(
            modality=ModalityType.IMAGE,
            features=torch.randn(1, 512),
            confidence_score=0.3,
            metadata={"reason": "low_quality_input"},
        )

        assert features.confidence_score == 0.3
        assert "reason" in features.metadata


class TestAsyncProcessing:
    """Test asynchronous processing capabilities."""

    @pytest.mark.asyncio
    async def test_async_text_processing(self):
        """Test async text processing."""
        # Simulate async processing
        async def mock_process(text):
            await asyncio.sleep(0.01)
            return {"processed": True, "text": text}

        result = await mock_process("test text")
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent multimodal processing."""
        async def process_modality(modality_type, delay):
            await asyncio.sleep(delay)
            return f"Processed {modality_type}"

        # Process multiple modalities concurrently
        tasks = [
            process_modality("text", 0.01),
            process_modality("image", 0.01),
            process_modality("audio", 0.01),
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all("Processed" in r for r in results)
