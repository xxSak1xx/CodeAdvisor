# src/configls.py

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/dir.config.yaml") -> Dict[str, Any]:
    """
    从 YAML 文件加载目录配置。
    如果文件不存在，则返回包含默认值的配置字典。
    """
    default_config = {
        "input_dir": ".",
        "ignore_dir": [".git", "node_modules", ".venv"],
        "output_dir": "./output",
        "model1": "gpt-3.5-turbo",
        "model2": "gpt-4",
        "model_embed": "text-embedding-3-small",
        "base_url1": None,
        "base_url2": None,
        "base_url_embed": None,
        "reindex": True,
        "persist_dir": "./data/chroma",
        "chunk_size": 2000,
        "chunk_overlap": 200,
        "top_k": 4,
        "vector_db": "chroma"
    }
    
    path = Path(config_path)
    if not path.exists():
        return default_config
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        if not config:
            return default_config
        
        # Merge loaded config with defaults to ensure all keys exist
        merged_config = default_config.copy()
        merged_config.update(config)
        return merged_config

def save_config(config: Dict[str, Any], config_path: str = "config/dir.config.yaml"):
    """
    将配置保存到 YAML 文件中。
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
