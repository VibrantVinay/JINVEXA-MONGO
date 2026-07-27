import os
from dotenv import load_dotenv
from typing import List, Optional, Dict

class Config:
    """
    Configuration manager with model selection capabilities.
    Predefined model list for easy switching.
    """

    # Predefined models with their descriptions and capabilities
    AVAILABLE_MODELS: Dict[str, Dict] = {
        # Vision Models (Support Document Scan/OCR)
        "gemma4:31b-cloud": {
            "description": "Gemma4 31B - Google's open model with vision support",
            "capabilities": ["text", "vision", "ocr", "document_scan"],
            "type": "vision"
        },
        "minimax-m3:cloud": {
            "description": "MiniMax M3 - MiniMax model with vision support",
            "capabilities": ["text", "vision", "ocr", "document_scan"],
            "type": "vision"
        },
        
        # Text-Only Models
        "gpt-oss:20b-cloud": {
            "description": "GPT-OSS 20B - Cloud optimized, text-only",
            "capabilities": ["text"],
            "type": "text"
        },
        "gpt-oss:120b-cloud": {
            "description": "GPT-OSS 120B - Large cloud model, text-only",
            "capabilities": ["text"],
            "type": "text"
        },
        "nemotron-3-super:cloud": {
            "description": "Nemotron 3 Super - NVIDIA's model, text-only",
            "capabilities": ["text"],
            "type": "text"
        },
    }

    # Vision models that support document scanning
    VISION_MODELS = ["gemma4:31b-cloud", "minimax-m3:cloud"]

    def __init__(self):
        load_dotenv()
        self.model = os.getenv("MODEL", "gemma4:31b-cloud")
        
        # Ensure model exists in available list
        if self.model not in self.AVAILABLE_MODELS:
            first_model = list(self.AVAILABLE_MODELS.keys())[0]
            print(f"⚠️ Model '{self.model}' not in available list. Using '{first_model}'")
            self.model = first_model

    def get_model(self) -> str:
        """Get the current model name."""
        return self.model

    def set_model(self, model_name: str) -> bool:
        """Set a new model if it exists in available list."""
        if model_name in self.AVAILABLE_MODELS:
            self.model = model_name
            self._update_env_model(model_name)
            return True
        return False

    def get_available_models(self) -> List[tuple]:
        """Get list of available models with names and descriptions."""
        return [(key, self.AVAILABLE_MODELS[key]["description"]) for key in self.AVAILABLE_MODELS]

    def get_model_description(self, model_name: str) -> Optional[str]:
        """Get description of a model."""
        if model_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_name]["description"]
        return None

    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities of a model."""
        if model_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_name].get("capabilities", ["text"])
        return ["text"]

    def get_model_type(self, model_name: str) -> str:
        """Get type of model (vision/text)."""
        if model_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_name].get("type", "text")
        return "text"

    def supports_document_scan(self, model_name: str = None) -> bool:
        """Check if the model supports document scanning/OCR."""
        if model_name is None:
            model_name = self.model
        return model_name in self.VISION_MODELS

    def get_vision_models(self) -> List[str]:
        """Get list of vision models (support document scan)."""
        return self.VISION_MODELS

    def get_text_models(self) -> List[str]:
        """Get list of text-only models."""
        return [m for m in self.AVAILABLE_MODELS if m not in self.VISION_MODELS]

    def _update_env_model(self, model_name: str):
        """Update the .env file with new model selection."""
        try:
            env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                model_found = False
                for i, line in enumerate(lines):
                    if line.startswith('MODEL='):
                        lines[i] = f'MODEL={model_name}\n'
                        model_found = True
                        break
                
                if not model_found:
                    lines.append(f'MODEL={model_name}\n')
                
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
        except Exception as e:
            print(f"⚠️ Could not update .env file: {e}")
# D:\Jinvexa\Config\Config.py

import os
from dotenv import load_dotenv
from typing import List, Optional, Dict


class Config:
    """
    Configuration manager with model selection capabilities.
    Predefined model list for easy switching.
    """

    # Predefined models with their descriptions and capabilities
    AVAILABLE_MODELS: Dict[str, Dict] = {
        # Vision Models (Support Document Scan/OCR)
        "gemma4:31b-cloud": {
            "description": "Gemma4 31B - Google's open model with vision support",
            "capabilities": ["text", "vision", "ocr", "document_scan"],
            "type": "vision"
        },
        "minimax-m3:cloud": {
            "description": "MiniMax M3 - MiniMax model with vision support",
            "capabilities": ["text", "vision", "ocr", "document_scan"],
            "type": "vision"
        },
        
        # Text-Only Models
        "gpt-oss:20b-cloud": {
            "description": "GPT-OSS 20B - Cloud optimized, text-only",
            "capabilities": ["text"],
            "type": "text"
        },
        "gpt-oss:120b-cloud": {
            "description": "GPT-OSS 120B - Large cloud model, text-only",
            "capabilities": ["text"],
            "type": "text"
        },
        "nemotron-3-super:cloud": {
            "description": "Nemotron 3 Super - NVIDIA's model, text-only",
            "capabilities": ["text"],
            "type": "text"
        },
    }

    # Vision models that support document scanning
    VISION_MODELS = ["gemma4:31b-cloud", "minimax-m3:cloud"]

    def __init__(self):
        load_dotenv()
        self.model = os.getenv("MODEL", "gemma4:31b-cloud")
        
        # Ensure model exists in available list
        if self.model not in self.AVAILABLE_MODELS:
            first_model = list(self.AVAILABLE_MODELS.keys())[0]
            print(f"⚠️ Model '{self.model}' not in available list. Using '{first_model}'")
            self.model = first_model

    def get_model(self) -> str:
        """Get the current model name."""
        return self.model

    def set_model(self, model_name: str) -> bool:
        """Set a new model if it exists in available list."""
        if model_name in self.AVAILABLE_MODELS:
            self.model = model_name
            self._update_env_model(model_name)
            return True
        return False

    def get_available_models(self) -> List[tuple]:
        """Get list of available models with names and descriptions."""
        return [(key, self.AVAILABLE_MODELS[key]["description"]) for key in self.AVAILABLE_MODELS]

    def get_model_description(self, model_name: str) -> Optional[str]:
        """Get description of a model."""
        if model_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_name]["description"]
        return None

    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities of a model."""
        if model_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_name].get("capabilities", ["text"])
        return ["text"]

    def get_model_type(self, model_name: str) -> str:
        """Get type of model (vision/text)."""
        if model_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_name].get("type", "text")
        return "text"

    def supports_document_scan(self, model_name: str = None) -> bool:
        """Check if the model supports document scanning/OCR."""
        if model_name is None:
            model_name = self.model
        return model_name in self.VISION_MODELS

    def get_vision_models(self) -> List[str]:
        """Get list of vision models (support document scan)."""
        return self.VISION_MODELS

    def get_text_models(self) -> List[str]:
        """Get list of text-only models."""
        return [m for m in self.AVAILABLE_MODELS if m not in self.VISION_MODELS]

    def _update_env_model(self, model_name: str):
        """Update the .env file with new model selection."""
        try:
            env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                model_found = False
                for i, line in enumerate(lines):
                    if line.startswith('MODEL='):
                        lines[i] = f'MODEL={model_name}\n'
                        model_found = True
                        break
                
                if not model_found:
                    lines.append(f'MODEL={model_name}\n')
                
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
        except Exception as e:
            print(f"⚠️ Could not update .env file: {e}")

    def get_storage_type(self) -> str:
        """Get the current storage backend type ('json' or 'mongodb')."""
        return os.getenv("STORAGE_TYPE", "json").lower()

    def get_mongo_uri(self) -> str:
        """Get the MongoDB connection URI from environment or default to local."""
        return os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    def get_mongo_db_name(self) -> str:
        """Get the target MongoDB database name."""
        return os.getenv("MONGODB_DB_NAME", "jinvexa_db")
