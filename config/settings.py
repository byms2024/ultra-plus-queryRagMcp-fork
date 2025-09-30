#!/usr/bin/env python3
"""
Centralized configuration settings for the Generic RAG system.
All environment variables and settings are consolidated here.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .profiles.profile_factory import ProfileFactory
from .profiles.base_profile import BaseProfile

# =============================================================================
# PROFILE SELECTION - Change this to switch between profiles
# =============================================================================
PROFILE = "default_profile"  # Options: "default_profile"

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.resolve()

# API Configuration (for backward compatibility)
GENERATION_MODEL = "gemini-1.5-flash"
API_PORT = 8000

@dataclass
class Config:
    """Text2Query configuration class for backward compatibility."""
    google_api_key: str
    generation_model: str
    port: int
    profile_name: str

@dataclass
class SystemConfig:
    """System configuration with all settings consolidated."""
    
    # Profile Configuration
    profile_name: str = PROFILE
    
    # LLM Settings
    generation_model: str = "gemini-1.5-flash"
    embedding_model: str = "gemini-embedding-001"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Google API Key (now loaded from profile-specific config_api_keys.py)
    google_api_key: str = "PLACEHOLDER"  # This will be overridden by profile-specific config
    
    # Vector Store Settings
    vector_store_type: str = "chroma"
    vector_store_path: str = str(BASE_DIR / "storage" / "vector_store")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG Settings
    top_k: int = 50
    max_iterations: int = 10
    
    # API Settings
    api_port: int = 8000
    mcp_port: int = 7800
    
    # Data Settings
    sample_size: Optional[int] = None
    
    # Optional: LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "generic-rag-system"
    enable_tracing: bool = False
    
    # Logging Settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


def load_config() -> Config:
    """Load Text2Query configuration for backward compatibility."""
    return Config(
        google_api_key="PLACEHOLDER",  # Will be loaded dynamically via provider config
        generation_model=GENERATION_MODEL,
        port=API_PORT,
        profile_name=PROFILE,
    )

def load_profile(config: Config) -> 'DataProfile':
    """Load the appropriate data profile based on configuration."""
    try:
        return ProfileFactory.create_profile(config.profile_name)
    except (ValueError, ImportError) as e:
        from config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.warning(f"Failed to load profile '{config.profile_name}': {e}")
        logger.info("Falling back to default profile")
        return ProfileFactory.get_default_profile()

def load_system_config() -> SystemConfig:
    """Load system configuration."""
    return SystemConfig()


def get_profile() -> BaseProfile:
    """Get the configured profile."""
    config = load_system_config()
    return ProfileFactory.create_profile(config.profile_name)


def get_data_file_path() -> str:
    """Get the data file path from the current profile."""
    profile = get_profile()
    return profile.get_data_file_path()


def get_vector_store_path() -> str:
    """Get the vector store path."""
    config = load_system_config()
    return config.vector_store_path


def get_api_port() -> int:
    """Get the API port."""
    config = load_system_config()
    return config.api_port


def get_mcp_port() -> int:
    """Get the MCP port."""
    config = load_system_config()
    return config.mcp_port


def get_google_api_key() -> str:
    """Get the Google API key from the active profile's config_api_keys.py."""
    try:
        # Get the active profile
        profile = get_profile()
        
        # Use generic import - no hardcoded profile names
        try:
            module_path = f"config.profiles.{profile.profile_name}.config_api_keys"
            mod = __import__(module_path, fromlist=['GCP_API_KEY'])
            GCP_API_KEY = getattr(mod, 'GCP_API_KEY')
        except Exception:
            raise ValueError(f"Unknown profile: {profile.profile_name}")
        
        return GCP_API_KEY
    except ImportError as e:
        # Fallback to system config if profile-specific config not found
        config = load_system_config()
        return config.google_api_key


def get_generation_model() -> str:
    """Get the generation model."""
    config = load_system_config()
    return config.generation_model


def get_embedding_model() -> str:
    """Get the embedding model."""
    config = load_system_config()
    return config.embedding_model


def get_temperature() -> float:
    """Get the temperature setting."""
    config = load_system_config()
    return config.temperature


def get_max_tokens() -> int:
    """Get the max tokens setting."""
    config = load_system_config()
    return config.max_tokens


def get_chunk_size() -> int:
    """Get the chunk size."""
    config = load_system_config()
    return config.chunk_size


def get_chunk_overlap() -> int:
    """Get the chunk overlap."""
    config = load_system_config()
    return config.chunk_overlap


def get_top_k() -> int:
    """Get the top_k setting."""
    config = load_system_config()
    return config.top_k


def get_max_iterations() -> int:
    """Get the max iterations."""
    config = load_system_config()
    return config.max_iterations


def get_sample_size() -> Optional[int]:
    """Get the sample size."""
    config = load_system_config()
    return config.sample_size


def get_langsmith_api_key() -> Optional[str]:
    """Get the LangSmith API key."""
    config = load_system_config()
    return config.langsmith_api_key


def get_langsmith_project() -> str:
    """Get the LangSmith project."""
    config = load_system_config()
    return config.langsmith_project


def get_enable_tracing() -> bool:
    """Get the enable tracing setting."""
    config = load_system_config()
    return config.enable_tracing


def get_log_level() -> str:
    """Get the log level."""
    config = load_system_config()
    return config.log_level


def get_log_to_file() -> bool:
    """Get the log to file setting."""
    config = load_system_config()
    return config.log_to_file


def get_log_to_console() -> bool:
    """Get the log to console setting."""
    config = load_system_config()
    return config.log_to_console


def get_max_file_size() -> int:
    """Get the max file size."""
    config = load_system_config()
    return config.max_file_size


def get_backup_count() -> int:
    """Get the backup count."""
    config = load_system_config()
    return config.backup_count


# Backward compatibility aliases
def load_langchain_config():
    """Backward compatibility function."""
    return load_system_config()


class LangChainConfig:
    """Backward compatibility class."""
    
    def __init__(self):
        config = load_system_config()
        profile = get_profile()
        
        # Map to old interface
        self.generation_model = config.generation_model
        self.embedding_model = config.embedding_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.vector_store_type = config.vector_store_type
        self.vector_store_path = config.vector_store_path
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.top_k = config.top_k
        self.max_iterations = config.max_iterations
        self.mcp_port = config.mcp_port
        self.api_port = config.api_port
        self.csv_file = profile.get_data_file_path()
        self.sample_size = config.sample_size
        self.google_api_key = config.google_api_key
        self.langsmith_api_key = config.langsmith_api_key
        self.langsmith_project = config.langsmith_project
        self.enable_tracing = config.enable_tracing
