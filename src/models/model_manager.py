"""
Model Manager - Centralized ML model loading and caching
"""

import os
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from ..utils.logger import get_logger

# Load environment variables
load_dotenv()

class ModelManager:
    """Centralized manager for ML model loading and caching"""
    
    def __init__(self, 
                 device: Optional[str] = None,
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model manager
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            cache_dir: Directory to cache models (None for default)
        """
        self.logger = get_logger().bind_context(service="model_manager")
        
        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info("model_manager_initialized", device=self.device)
        
        # Cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ecg-audio-analyzer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
        
        # Hugging Face token
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            self.logger.warning("hf_token_not_found", 
                              message="HF_TOKEN not found, some models may not be accessible")
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available() and "cuda" in self.device
    
    def clear_cache(self):
        """Clear model cache"""
        self.logger.info("clearing_model_cache")
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model manager information"""
        return {
            "device": self.device,
            "gpu_available": self.is_gpu_available(),
            "cached_models": list(self._model_cache.keys()),
            "cache_dir": str(self.cache_dir),
            "hf_token_available": bool(self.hf_token)
        }
    
    def _cache_model(self, model_key: str, model: Any) -> Any:
        """Cache a model"""
        self._model_cache[model_key] = model
        self.logger.info("model_cached", model_key=model_key)
        return model
    
    def _get_cached_model(self, model_key: str) -> Optional[Any]:
        """Get cached model"""
        return self._model_cache.get(model_key)
    
    def load_speaker_model(self, model_name: str = "pyannote/speaker-diarization-3.1") -> Any:
        """
        Load speaker diarization model
        
        Args:
            model_name: Hugging Face model name
            
        Returns:
            Loaded pyannote pipeline
        """
        model_key = f"speaker_{model_name}"
        
        # Check cache first
        cached_model = self._get_cached_model(model_key)
        if cached_model is not None:
            self.logger.info("using_cached_speaker_model", model_name=model_name)
            return cached_model
        
        self.logger.info("loading_speaker_model", model_name=model_name)
        
        try:
            from pyannote.audio import Pipeline
            
            # Load pipeline with authentication
            if self.hf_token:
                pipeline = Pipeline.from_pretrained(
                    model_name, 
                    use_auth_token=self.hf_token
                )
            else:
                pipeline = Pipeline.from_pretrained(model_name)
            
            # Move to device
            if hasattr(pipeline, 'to'):
                pipeline = pipeline.to(torch.device(self.device))
            
            self.logger.info("speaker_model_loaded", 
                           model_name=model_name, 
                           device=self.device)
            
            return self._cache_model(model_key, pipeline)
            
        except Exception as e:
            self.logger.error("speaker_model_load_failed", 
                            model_name=model_name, 
                            error=str(e))
            raise
    
    def load_emotion_model(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base") -> tuple:
        """
        Load emotion analysis model
        
        Args:
            model_name: Hugging Face model name
            
        Returns:
            Tuple of (tokenizer, model)
        """
        model_key = f"emotion_{model_name}"
        
        # Check cache first
        cached_model = self._get_cached_model(model_key)
        if cached_model is not None:
            self.logger.info("using_cached_emotion_model", model_name=model_name)
            return cached_model
        
        self.logger.info("loading_emotion_model", model_name=model_name)
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=self.hf_token if self.hf_token else None
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                use_auth_token=self.hf_token if self.hf_token else None
            )
            
            # Move model to device
            model = model.to(torch.device(self.device))
            model.eval()  # Set to evaluation mode
            
            self.logger.info("emotion_model_loaded", 
                           model_name=model_name, 
                           device=self.device)
            
            return self._cache_model(model_key, (tokenizer, model))
            
        except Exception as e:
            self.logger.error("emotion_model_load_failed", 
                            model_name=model_name, 
                            error=str(e))
            raise
    
    def load_whisperx_model(self, 
                           model_size: str = "base",
                           compute_type: str = "float16",
                           language: Optional[str] = None) -> Any:
        """
        Load WhisperX model
        
        Args:
            model_size: WhisperX model size (tiny, base, small, medium, large-v2)
            compute_type: Compute type (float16, float32, int8)
            language: Language code (None for auto-detection)
            
        Returns:
            Loaded WhisperX model
        """
        model_key = f"whisperx_{model_size}_{compute_type}_{language}"
        
        # Check cache first
        cached_model = self._get_cached_model(model_key)
        if cached_model is not None:
            self.logger.info("using_cached_whisperx_model", 
                           model_size=model_size,
                           compute_type=compute_type)
            return cached_model
        
        self.logger.info("loading_whisperx_model", 
                        model_size=model_size,
                        compute_type=compute_type,
                        language=language)
        
        try:
            import whisperx
            
            # Use CPU compute type if on CPU
            if self.device == "cpu":
                compute_type = "float32"
            
            # Load WhisperX model
            model = whisperx.load_model(
                model_size,
                device=self.device,
                compute_type=compute_type,
                language=language
            )
            
            self.logger.info("whisperx_model_loaded", 
                           model_size=model_size, 
                           device=self.device,
                           compute_type=compute_type)
            
            return self._cache_model(model_key, model)
            
        except Exception as e:
            self.logger.error("whisperx_model_load_failed", 
                            model_size=model_size, 
                            error=str(e))
            raise
    
    def load_whisperx_alignment_model(self, language_code: str) -> tuple:
        """
        Load WhisperX alignment model for better word-level timestamps
        
        Args:
            language_code: Language code for alignment model
            
        Returns:
            Tuple of (alignment_model, metadata)
        """
        model_key = f"whisperx_align_{language_code}"
        
        # Check cache first
        cached_model = self._get_cached_model(model_key)
        if cached_model is not None:
            self.logger.info("using_cached_alignment_model", language=language_code)
            return cached_model
        
        self.logger.info("loading_alignment_model", language=language_code)
        
        try:
            import whisperx
            
            # Load alignment model
            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            
            self.logger.info("alignment_model_loaded", 
                           language=language_code,
                           device=self.device)
            
            return self._cache_model(model_key, (model, metadata))
            
        except Exception as e:
            self.logger.error("alignment_model_load_failed", 
                            language=language_code,
                            error=str(e))
            raise


# Global model manager instance
_global_model_manager: Optional[ModelManager] = None


def get_model_manager(**kwargs) -> ModelManager:
    """Get or create global model manager instance"""
    global _global_model_manager
    
    if _global_model_manager is None:
        _global_model_manager = ModelManager(**kwargs)
    
    return _global_model_manager


def setup_models(device: Optional[str] = None, cache_dir: Optional[str] = None) -> ModelManager:
    """Setup global model manager"""
    global _global_model_manager
    
    _global_model_manager = ModelManager(device=device, cache_dir=cache_dir)
    return _global_model_manager