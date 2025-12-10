"""
Multimodal Processing for GRPO Agent Framework

This module provides comprehensive multimodal capabilities for processing and fusing
different types of inputs including text, images, audio, video, and structured data.
"""

import asyncio
import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

# Lazy imports for transformers to avoid torch/torchvision compatibility issues
_transformers_loaded = False
AutoModel = None
AutoProcessor = None
AutoTokenizer = None

def _load_transformers():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_loaded, AutoModel, AutoProcessor, AutoTokenizer
    if _transformers_loaded:
        return True
    try:
        from transformers import AutoModel as _AutoModel
        from transformers import AutoProcessor as _AutoProcessor
        from transformers import AutoTokenizer as _AutoTokenizer
        AutoModel = _AutoModel
        AutoProcessor = _AutoProcessor
        AutoTokenizer = _AutoTokenizer
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import librosa
    import soundfile as sf

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .error_handling import ErrorHandler, RetryConfig, retry_async
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of input modalities"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"  # JSON, CSV, etc.
    SPEECH = "speech"
    GESTURE = "gesture"
    SENSOR = "sensor"  # IoT sensor data
    MULTIMODAL = "multimodal"  # Combined modalities


class FusionStrategy(Enum):
    """Strategies for multimodal fusion"""

    EARLY_FUSION = "early_fusion"  # Combine raw features
    LATE_FUSION = "late_fusion"  # Combine after individual processing
    HYBRID_FUSION = "hybrid_fusion"  # Combination of early and late
    ATTENTION_FUSION = "attention_fusion"  # Attention-based fusion
    CROSS_MODAL_ATTENTION = (
        "cross_modal_attention"  # Cross-attention between modalities
    )
    TENSOR_FUSION = "tensor_fusion"  # Tensor-based fusion


class ProcessingQuality(Enum):
    """Quality levels for processing"""

    LOW = "low"  # Fast, less accurate
    MEDIUM = "medium"  # Balanced
    HIGH = "high"  # Slow, high accuracy
    ADAPTIVE = "adaptive"  # Quality based on content complexity


@dataclass
class ModalityInput:
    """Input data for a specific modality"""

    modality: ModalityType
    data: Any  # Raw data (bytes, numpy array, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    encoding: Optional[str] = None  # base64, raw, etc.
    source: Optional[str] = None  # URL, file path, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    quality: ProcessingQuality = ProcessingQuality.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "metadata": self.metadata,
            "encoding": self.encoding,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality.value,
            "data_type": type(self.data).__name__,
            "data_size": len(self.data) if hasattr(self.data, "__len__") else 0,
        }


@dataclass
class ProcessedFeatures:
    """Processed features from a modality"""

    modality: ModalityType
    features: torch.Tensor  # Processed feature tensor
    attention_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "feature_shape": list(self.features.shape)
            if self.features is not None
            else None,
            "has_attention_mask": self.attention_mask is not None,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "confidence_score": self.confidence_score,
        }


class ModalityProcessor(ABC):
    """Abstract base class for modality processors"""

    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.error_handler = ErrorHandler()

    @abstractmethod
    async def process(self, input_data: ModalityInput) -> ProcessedFeatures:
        """Process input data and return features"""
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        """Get the dimension of output features"""
        pass

    @abstractmethod
    async def initialize(self):
        """Initialize the processor"""
        pass


class TextProcessor(ModalityProcessor):
    """Text processing using transformer models"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        super().__init__(model_name, device)
        self.tokenizer = None
        self.model = None
        self.max_length = 512

    async def initialize(self):
        """Initialize text processor"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers required for text processing")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Initialized text processor with {self.model_name}")
        except Exception as e:
            self.error_handler.handle_error(e, "text_processor", "initialize")
            raise

    async def process(self, input_data: ModalityInput) -> ProcessedFeatures:
        """Process text input"""
        start_time = datetime.now()

        try:
            text = input_data.data
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token or mean pooling
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    features = outputs.pooler_output
                else:
                    # Mean pooling
                    features = outputs.last_hidden_state.mean(dim=1)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessedFeatures(
                modality=ModalityType.TEXT,
                features=features,
                attention_mask=inputs.get("attention_mask"),
                metadata={
                    "text_length": len(text),
                    "token_count": inputs["input_ids"].shape[1],
                    "model": self.model_name,
                },
                processing_time=processing_time,
                confidence_score=1.0,
            )

        except Exception as e:
            self.error_handler.handle_error(e, "text_processor", "process")
            # Return zero features on error
            return ProcessedFeatures(
                modality=ModalityType.TEXT,
                features=torch.zeros(1, self.get_feature_dimension()),
                processing_time=0.0,
                confidence_score=0.0,
            )

    def get_feature_dimension(self) -> int:
        """Get text feature dimension"""
        if self.model is not None:
            return self.model.config.hidden_size
        return 384  # Default for MiniLM


class ImageProcessor(ModalityProcessor):
    """Image processing using vision models"""

    def __init__(
        self, model_name: str = "google/vit-base-patch16-224", device: str = "cpu"
    ):
        super().__init__(model_name, device)
        self.processor = None
        self.model = None

    async def initialize(self):
        """Initialize image processor"""
        if not TORCH_AVAILABLE or not PIL_AVAILABLE:
            raise ImportError(
                "PyTorch, transformers, and PIL required for image processing"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Initialized image processor with {self.model_name}")
        except Exception as e:
            self.error_handler.handle_error(e, "image_processor", "initialize")
            raise

    async def process(self, input_data: ModalityInput) -> ProcessedFeatures:
        """Process image input"""
        start_time = datetime.now()

        try:
            # Handle different image input formats
            if isinstance(input_data.data, str):
                if input_data.encoding == "base64":
                    image_bytes = base64.b64decode(input_data.data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    image = Image.open(input_data.data)  # File path
            elif isinstance(input_data.data, bytes):
                image = Image.open(io.BytesIO(input_data.data))
            elif isinstance(input_data.data, np.ndarray):
                image = Image.fromarray(input_data.data)
            else:
                image = input_data.data  # Already PIL Image

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Process image
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use pooler output or CLS token
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0]  # CLS token

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessedFeatures(
                modality=ModalityType.IMAGE,
                features=features,
                metadata={
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "model": self.model_name,
                },
                processing_time=processing_time,
                confidence_score=1.0,
            )

        except Exception as e:
            self.error_handler.handle_error(e, "image_processor", "process")
            return ProcessedFeatures(
                modality=ModalityType.IMAGE,
                features=torch.zeros(1, self.get_feature_dimension()),
                processing_time=0.0,
                confidence_score=0.0,
            )

    def get_feature_dimension(self) -> int:
        """Get image feature dimension"""
        if self.model is not None:
            return self.model.config.hidden_size
        return 768  # Default for ViT


class AudioProcessor(ModalityProcessor):
    """Audio processing using audio models"""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = "cpu"):
        super().__init__(model_name, device)
        self.processor = None
        self.model = None
        self.sample_rate = 16000

    async def initialize(self):
        """Initialize audio processor"""
        if not TORCH_AVAILABLE or not AUDIO_AVAILABLE:
            raise ImportError(
                "PyTorch, transformers, and librosa required for audio processing"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Initialized audio processor with {self.model_name}")
        except Exception as e:
            self.error_handler.handle_error(e, "audio_processor", "initialize")
            raise

    async def process(self, input_data: ModalityInput) -> ProcessedFeatures:
        """Process audio input"""
        start_time = datetime.now()

        try:
            # Handle different audio input formats
            if isinstance(input_data.data, str):
                # File path
                audio, sr = librosa.load(input_data.data, sr=self.sample_rate)
            elif isinstance(input_data.data, bytes):
                # Audio bytes
                audio, sr = sf.read(io.BytesIO(input_data.data))
                if sr != self.sample_rate:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=self.sample_rate
                    )
            else:
                # Already numpy array
                audio = input_data.data

            # Ensure mono channel
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            # Process audio
            inputs = self.processor(
                audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling over time dimension
                features = outputs.last_hidden_state.mean(dim=1)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessedFeatures(
                modality=ModalityType.AUDIO,
                features=features,
                metadata={
                    "audio_length": len(audio),
                    "sample_rate": self.sample_rate,
                    "duration": len(audio) / self.sample_rate,
                    "model": self.model_name,
                },
                processing_time=processing_time,
                confidence_score=1.0,
            )

        except Exception as e:
            self.error_handler.handle_error(e, "audio_processor", "process")
            return ProcessedFeatures(
                modality=ModalityType.AUDIO,
                features=torch.zeros(1, self.get_feature_dimension()),
                processing_time=0.0,
                confidence_score=0.0,
            )

    def get_feature_dimension(self) -> int:
        """Get audio feature dimension"""
        if self.model is not None:
            return self.model.config.hidden_size
        return 768  # Default for Wav2Vec2


class StructuredDataProcessor(ModalityProcessor):
    """Process structured data (JSON, tables, etc.)"""

    def __init__(self, embedding_dim: int = 512, device: str = "cpu"):
        super().__init__("structured_processor", device)
        self.embedding_dim = embedding_dim
        self.encoder = None

    async def initialize(self):
        """Initialize structured data processor"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for structured data processing")

        # Simple MLP encoder for structured data
        self.encoder = nn.Sequential(
            nn.Linear(1, 128),  # Will be dynamically sized
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_dim),
        ).to(self.device)
        logger.info("Initialized structured data processor")

    async def process(self, input_data: ModalityInput) -> ProcessedFeatures:
        """Process structured data input"""
        start_time = datetime.now()

        try:
            data = input_data.data

            # Convert to tensor
            if isinstance(data, dict):
                # Simple approach: flatten dict values
                values = self._flatten_dict(data)
                tensor = torch.tensor(values, dtype=torch.float32)
            elif isinstance(data, list):
                tensor = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).float()
            else:
                # Try to convert to tensor
                tensor = torch.tensor([float(data)], dtype=torch.float32)

            # Reshape for batch processing
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)

            # Dynamically adjust encoder input size if needed
            input_size = tensor.shape[-1]
            if self.encoder[0].in_features != input_size:
                self.encoder[0] = nn.Linear(input_size, 128).to(self.device)

            tensor = tensor.to(self.device)

            # Get features
            with torch.no_grad():
                features = self.encoder(tensor)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessedFeatures(
                modality=ModalityType.STRUCTURED,
                features=features,
                metadata={"input_size": input_size, "data_type": type(data).__name__},
                processing_time=processing_time,
                confidence_score=1.0,
            )

        except Exception as e:
            self.error_handler.handle_error(e, "structured_processor", "process")
            return ProcessedFeatures(
                modality=ModalityType.STRUCTURED,
                features=torch.zeros(1, self.embedding_dim),
                processing_time=0.0,
                confidence_score=0.0,
            )

    def _flatten_dict(
        self, data: dict, parent_key: str = "", sep: str = "_"
    ) -> List[float]:
        """Flatten nested dictionary to list of numbers"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep))
            elif isinstance(v, (int, float)):
                items.append(float(v))
            elif isinstance(v, str):
                # Simple string to number conversion (length)
                items.append(float(len(v)))
            elif isinstance(v, bool):
                items.append(float(v))
            elif isinstance(v, list):
                items.extend(
                    [
                        float(x) if isinstance(x, (int, float)) else len(str(x))
                        for x in v
                    ]
                )
        return items

    def get_feature_dimension(self) -> int:
        """Get structured data feature dimension"""
        return self.embedding_dim


class MultimodalFusion(nn.Module):
    """Neural network for fusing multimodal features"""

    def __init__(
        self,
        modality_dims: Dict[ModalityType, int],
        fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION,
        output_dim: int = 512,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.fusion_strategy = fusion_strategy
        self.output_dim = output_dim

        # Project all modalities to same dimension
        self.projectors = nn.ModuleDict(
            {
                modality.value: nn.Linear(dim, output_dim)
                for modality, dim in modality_dims.items()
            }
        )

        if fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            self.attention = nn.MultiheadAttention(
                output_dim, num_heads=8, batch_first=True
            )
            self.fusion_layer = nn.Linear(output_dim, output_dim)
        elif fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            self.cross_attention = nn.ModuleDict(
                {
                    f"{mod1.value}_{mod2.value}": nn.MultiheadAttention(
                        output_dim, num_heads=4, batch_first=True
                    )
                    for mod1 in modality_dims.keys()
                    for mod2 in modality_dims.keys()
                    if mod1 != mod2
                }
            )
        elif fusion_strategy == FusionStrategy.TENSOR_FUSION:
            # Simplified tensor fusion
            total_dim = sum(modality_dims.values())
            self.tensor_fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim),
            )
        else:
            # Simple concatenation + MLP
            total_dim = len(modality_dims) * output_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
            )

    def forward(
        self, modality_features: Dict[ModalityType, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse features from multiple modalities"""
        # Project all features to same dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if modality.value in self.projectors:
                projected_features[modality] = self.projectors[modality.value](features)

        if self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            # Stack features for attention
            feature_stack = torch.stack(list(projected_features.values()), dim=1)
            attended_features, _ = self.attention(
                feature_stack, feature_stack, feature_stack
            )
            fused = attended_features.mean(dim=1)
            return self.fusion_layer(fused)

        elif self.fusion_strategy == FusionStrategy.EARLY_FUSION:
            # Concatenate raw features
            concatenated = torch.cat(
                [features for features in projected_features.values()], dim=-1
            )
            return self.fusion_layer(concatenated)

        elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
            # Average projected features
            return torch.stack(list(projected_features.values())).mean(dim=0)

        else:
            # Default: concatenation + MLP
            concatenated = torch.cat(list(projected_features.values()), dim=-1)
            return self.fusion_layer(concatenated)


class MultimodalProcessor:
    """Main multimodal processing coordinator"""

    def __init__(
        self,
        fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION,
        device: str = "cpu",
    ):
        self.fusion_strategy = fusion_strategy
        self.device = device

        # Initialize processors
        self.processors: Dict[ModalityType, ModalityProcessor] = {}
        self.fusion_network = None

        self.error_handler = ErrorHandler()
        self.monitoring = get_monitoring_service()

    async def initialize(self, enabled_modalities: List[ModalityType]):
        """Initialize processors for enabled modalities"""
        for modality in enabled_modalities:
            try:
                if modality == ModalityType.TEXT:
                    processor = TextProcessor(device=self.device)
                elif modality == ModalityType.IMAGE:
                    processor = ImageProcessor(device=self.device)
                elif modality == ModalityType.AUDIO:
                    processor = AudioProcessor(device=self.device)
                elif modality == ModalityType.STRUCTURED:
                    processor = StructuredDataProcessor(device=self.device)
                else:
                    logger.warning(
                        f"Processor for {modality.value} not implemented yet"
                    )
                    continue

                await processor.initialize()
                self.processors[modality] = processor
                logger.info(f"Initialized {modality.value} processor")

            except Exception as e:
                self.error_handler.handle_error(
                    e, "multimodal_processor", f"initialize_{modality.value}"
                )

        # Initialize fusion network
        if len(self.processors) > 1:
            modality_dims = {
                modality: processor.get_feature_dimension()
                for modality, processor in self.processors.items()
            }
            self.fusion_network = MultimodalFusion(
                modality_dims, self.fusion_strategy
            ).to(self.device)
            logger.info(f"Initialized fusion network with {self.fusion_strategy.value}")

    @monitor_async_function("multimodal_processing")
    async def process_multimodal_input(
        self, inputs: List[ModalityInput]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process multimodal input and return fused features"""
        start_time = datetime.now()

        try:
            # Process each modality
            processed_features = {}
            processing_metadata = {}

            for input_data in inputs:
                if input_data.modality in self.processors:
                    features = await self.processors[input_data.modality].process(
                        input_data
                    )
                    processed_features[input_data.modality] = features.features
                    processing_metadata[input_data.modality.value] = features.to_dict()

            # Fuse features if multiple modalities
            if len(processed_features) > 1 and self.fusion_network is not None:
                fused_features = self.fusion_network(processed_features)
            elif len(processed_features) == 1:
                fused_features = list(processed_features.values())[0]
            else:
                # No valid features
                output_dim = 512  # Default
                fused_features = torch.zeros(1, output_dim, device=self.device)

            total_time = (datetime.now() - start_time).total_seconds()

            # Log metrics
            await self._log_processing_metrics(inputs, total_time)

            metadata = {
                "modalities_processed": [inp.modality.value for inp in inputs],
                "fusion_strategy": self.fusion_strategy.value,
                "total_processing_time": total_time,
                "feature_shape": list(fused_features.shape),
                "individual_processing": processing_metadata,
            }

            return fused_features, metadata

        except Exception as e:
            self.error_handler.handle_error(e, "multimodal_processor", "process")
            # Return zero features on error
            return torch.zeros(1, 512, device=self.device), {"error": str(e)}

    async def _log_processing_metrics(
        self, inputs: List[ModalityInput], processing_time: float
    ):
        """Log processing metrics"""
        metrics = {
            "processing_time": processing_time,
            "num_modalities": len(inputs),
            "modalities": [inp.modality.value for inp in inputs],
        }

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                await self.monitoring.record_metric(f"multimodal.{metric_name}", value)

    def get_supported_modalities(self) -> List[ModalityType]:
        """Get list of supported modalities"""
        return list(self.processors.keys())

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "supported_modalities": [mod.value for mod in self.processors.keys()],
            "fusion_strategy": self.fusion_strategy.value,
            "device": self.device,
            "processor_info": {
                modality.value: {
                    "model_name": processor.model_name,
                    "feature_dim": processor.get_feature_dimension(),
                }
                for modality, processor in self.processors.items()
            },
        }


# Factory functions
def create_multimodal_processor(
    modalities: List[str],
    fusion_strategy: str = "attention_fusion",
    device: str = "cpu",
) -> MultimodalProcessor:
    """Create a multimodal processor with specified modalities"""
    modality_types = [ModalityType(mod) for mod in modalities]
    fusion_strat = FusionStrategy(fusion_strategy)

    processor = MultimodalProcessor(fusion_strat, device)
    return processor


def create_modality_input(
    modality: str,
    data: Any,
    metadata: Dict[str, Any] = None,
    encoding: str = None,
    quality: str = "medium",
) -> ModalityInput:
    """Create a modality input object"""
    return ModalityInput(
        modality=ModalityType(modality),
        data=data,
        metadata=metadata or {},
        encoding=encoding,
        quality=ProcessingQuality(quality),
    )


# Utility functions
def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode audio file to base64 string"""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def validate_multimodal_input(inputs: List[ModalityInput]) -> List[str]:
    """Validate multimodal inputs and return any issues"""
    issues = []

    for i, inp in enumerate(inputs):
        if inp.data is None:
            issues.append(f"Input {i}: No data provided")

        if inp.modality == ModalityType.TEXT and not isinstance(inp.data, (str, bytes)):
            issues.append(f"Input {i}: Text data must be string or bytes")

        if inp.modality == ModalityType.IMAGE and inp.encoding == "base64":
            try:
                base64.b64decode(inp.data)
            except Exception:
                issues.append(f"Input {i}: Invalid base64 encoding for image")

    return issues
