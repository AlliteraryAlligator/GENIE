from typing import Dict, Optional
import yaml
from pydantic import BaseModel

class ModelSpec(BaseModel):
    prefix: str
    role: str
    type: str
    path: str
    
    gpus: Optional[int] = None
    time: Optional[str] = None
    tokens: Optional[int] = None
    numgen: Optional[int] = None
    memory: Optional[int] = None

class ModelConfiguration(BaseModel):
    models: Dict[str, ModelSpec]

def load_model_configs(filepath: str = "models.yaml") -> Dict[str, ModelSpec]:
    """Reads the YAML file, filters out anchor definitions, and validates schema."""
    with open(filepath, "r") as file:
        raw_data = yaml.safe_load(file)
    
    cleaned_data = {
        key: value 
        for key, value in raw_data.items() 
        if not key.startswith("default_") and key != "api_model"
    }
    
    validated_config = ModelConfiguration(models=cleaned_data)
    
    return validated_config.models