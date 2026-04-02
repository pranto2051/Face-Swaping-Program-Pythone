"""
Unified Model Loader System
Manages loading of InsightFace, DeepFaceLab, DeepFake, and Enhancement Models
"""
import os
import logging
from typing import Optional, Dict, Any
import torch
from pathlib import Path
import json
import hashlib
from urllib.request import urlretrieve
from tqdm import tqdm
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Model registry with download URLs and metadata
MODEL_REGISTRY = {
    'insightface': {
        'detector': {
            'name': 'buffalo_l',
            'type': 'detection',
            'size_mb': 326,
        },
        'recognition': {
            'name': 'arcface_r100_v1',
            'type': 'recognition',
            'size_mb': 167,
        }
    },
    'deepfacelab': {
        'model_file': 'DeepFaceLab_SAEHD.h5',
        'type': 'h5',
        'source': 'local',  # Requires manual download
    },
    'deepfake': {
        'first_order': {
            'name': 'checkpoint.tar',
            'type': 'checkpoint',
            'url': 'https://yandexcloud.net/d/1PWNEqp-B5gLpL7U-ZoV1Bj0YwZN4sEzu/checkpoint.tar.gz',
            'size_mb': 47,
        }
    },
    'enhancers': {
        'gfpgan': {
            'name': 'GFPGANv1.4.pth',
            'type': 'pth',
            'source': 'local',
        },
        'realesrgan': {
            'name': 'RealESRGAN_x2_compact.pth',
            'type': 'pth',
            'source': 'local',
        },
        'codeformer': {
            'name': 'codeformer.pth',
            'type': 'pth',
            'source': 'local',
        }
    }
}


class ModelLoader:
    """Unified model loader with caching and lazy loading"""
    
    def __init__(self, model_cache_dir: str = './models'):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.ort_session_options = ort.SessionOptions()
        self.ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
    def get_insightface_model(self, model_name: str = 'buffalo_l', providers: list = None):
        """Load InsightFace detection and recognition models"""
        cache_key = f'insightface_{model_name}'
        
        if cache_key in self.loaded_models:
            logger.info(f"✓ Using cached InsightFace model: {model_name}")
            return self.loaded_models[cache_key]
        
        try:
            import insightface
            
            logger.info(f"→ Loading InsightFace model: {model_name}")
            
            # Use ONNX CPU providers
            if providers is None:
                providers = ['CPUExecutionProvider']  # macOS compatible
            
            # Load face analysis model
            app = insightface.app.FaceAnalysis(
                name=model_name,
                providers=providers,
                root=str(self.model_cache_dir / 'insightface')
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = CPU
            
            self.loaded_models[cache_key] = app
            logger.info(f"✓ InsightFace {model_name} loaded successfully")
            return app
            
        except Exception as e:
            logger.error(f"✗ Failed to load InsightFace: {e}")
            raise
    
    def get_torch_model(self, model_name: str, model_path: str, class_fn=None) -> torch.nn.Module:
        """Load PyTorch models"""
        cache_key = f'torch_{model_name}'
        
        if cache_key in self.loaded_models:
            logger.info(f"✓ Using cached PyTorch model: {model_name}")
            return self.loaded_models[cache_key]
        
        try:
            logger.info(f"→ Loading PyTorch model: {model_name}")
            
            if not os.path.exists(model_path):
                logger.error(f"✗ Model not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if class_fn is not None:
                model = class_fn()
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            
            model = model.to(self.device)
            model.eval()
            
            self.loaded_models[cache_key] = model
            logger.info(f"✓ PyTorch model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"✗ Failed to load PyTorch model {model_name}: {e}")
            raise
    
    def get_gfpgan_model(self, model_version: str = '1.4') -> 'torch.nn.Module':
        """Load GFPGAN face enhancement model"""
        cache_key = f'gfpgan_{model_version}'
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            from gfpgan import GFPGANer
            
            logger.info(f"→ Loading GFPGAN v{model_version}")
            
            model_path = self.model_cache_dir / 'enhancers' / f'GFPGANv{model_version}.pth'
            
            if not model_path.exists():
                logger.warning(f"GFPGAN model not found at {model_path}")
                return None
            
            gfpgan = GFPGANer(
                scale=2,
                model_path=str(model_path),
                upscale=4,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            
            self.loaded_models[cache_key] = gfpgan
            logger.info(f"✓ GFPGAN v{model_version} loaded")
            return gfpgan
            
        except Exception as e:
            logger.error(f"✗ Failed to load GFPGAN: {e}")
            return None
    
    def get_realesrgan_model(self, model_name: str = 'RealESRGAN_x2_compact') -> 'torch.nn.Module':
        """Load Real-ESRGAN model for upscaling"""
        cache_key = f'realesrgan_{model_name}'
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            from realesrgan import RealESRGANer
            
            logger.info(f"→ Loading Real-ESRGAN: {model_name}")
            
            model_path = self.model_cache_dir / 'enhancers' / f'{model_name}.pth'
            
            if not model_path.exists():
                logger.warning(f"Real-ESRGAN model not found at {model_path}")
                return None
            
            upsampler = RealESRGANer(
                scale=2,
                model_path=str(model_path),
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device.type == 'cuda' else False
            )
            
            self.loaded_models[cache_key] = upsampler
            logger.info(f"✓ Real-ESRGAN loaded")
            return upsampler
            
        except Exception as e:
            logger.error(f"✗ Failed to load Real-ESRGAN: {e}")
            return None
    
    def get_codeformer_model(self):
        """Load CodeFormer face restoration model"""
        cache_key = 'codeformer'
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            from gfpgan.archs.codeformer_arch import CodeFormer
            
            logger.info("→ Loading CodeFormer")
            
            model_path = self.model_cache_dir / 'enhancers' / 'codeformer.pth'
            
            if not model_path.exists():
                logger.warning(f"CodeFormer model not found at {model_path}")
                return None
            
            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256'],
                fix_modules=['quantizer', 'after_norm_loss']
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'params_ema' in checkpoint:
                net.load_state_dict(checkpoint['params_ema'])
            else:
                net.load_state_dict(checkpoint)
            
            net = net.to(self.device)
            net.eval()
            
            self.loaded_models[cache_key] = net
            logger.info("✓ CodeFormer loaded")
            return net
            
        except Exception as e:
            logger.error(f"✗ Failed to load CodeFormer: {e}")
            return None
    
    def check_model_exists(self, model_type: str, model_name: str) -> bool:
        """Check if model file exists"""
        if model_type == 'insightface':
            return True  # InsightFace auto-downloads
        
        model_dir = self.model_cache_dir / model_type
        model_path = model_dir / model_name
        
        return model_path.exists()
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get model registry information"""
        return MODEL_REGISTRY.get(model_type, {})
    
    def clear_cache(self):
        """Unload all cached models to free memory"""
        logger.info("Clearing model cache...")
        self.loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def list_loaded_models(self) -> list:
        """List all currently loaded models"""
        return list(self.loaded_models.keys())


# Singleton instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader(model_cache_dir: str = './models') -> ModelLoader:
    """Get or create model loader singleton"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(model_cache_dir)
    return _model_loader
