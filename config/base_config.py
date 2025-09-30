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
# CONSTANTS - Default configuration values that can be overridden by profiles
# =============================================================================

# Profile Selection
PROFILE = "default_profile"  # Options: "default_profile", "customized_profile"

# LLM Configuration
DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4000

# Vector Store Configuration
DEFAULT_VECTOR_STORE_TYPE = "chroma"  # Options: "chroma", "faiss"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# RAG Configuration
DEFAULT_TOP_K = 50
DEFAULT_MAX_ITERATIONS = 10

# API Configuration
DEFAULT_API_PORT = 8000
DEFAULT_MCP_PORT = 7800

# Data Configuration
DEFAULT_SAMPLE_SIZE = None

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_TO_FILE = True
DEFAULT_LOG_TO_CONSOLE = True
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

# LangSmith Configuration (Optional)
DEFAULT_LANGSMITH_API_KEY = None
DEFAULT_LANGSMITH_PROJECT = "generic-rag-system"
DEFAULT_ENABLE_TRACING = False

# API Configuration (for backward compatibility)
GENERATION_MODEL = DEFAULT_GENERATION_MODEL
API_PORT = DEFAULT_API_PORT

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.resolve()

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
    generation_model: str = DEFAULT_GENERATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    
    # Google API Key (now loaded from profile-specific config_api_keys.py)
    google_api_key: str = "PLACEHOLDER"  # This will be overridden by profile-specific config
    
    # Vector Store Settings
    vector_store_type: str = DEFAULT_VECTOR_STORE_TYPE
    vector_store_path: str = str(BASE_DIR / "storage" / "vector_store")
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    
    # RAG Settings
    top_k: int = DEFAULT_TOP_K
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    
    # API Settings
    api_port: int = DEFAULT_API_PORT
    mcp_port: int = DEFAULT_MCP_PORT
    
    # Data Settings
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE
    
    # Optional: LangSmith
    langsmith_api_key: Optional[str] = DEFAULT_LANGSMITH_API_KEY
    langsmith_project: str = DEFAULT_LANGSMITH_PROJECT
    enable_tracing: bool = DEFAULT_ENABLE_TRACING
    
    # Logging Settings
    log_level: str = DEFAULT_LOG_LEVEL
    log_to_file: bool = DEFAULT_LOG_TO_FILE
    log_to_console: bool = DEFAULT_LOG_TO_CONSOLE
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    backup_count: int = DEFAULT_BACKUP_COUNT


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
    """Load system configuration with profile overrides."""
    # Start with default configuration
    config = SystemConfig()
    
    # Try to get profile-specific overrides
    try:
        profile = ProfileFactory.create_profile(config.profile_name)
        
        # Override with profile-specific constants if they exist
        profile_module = __import__(f"config.profiles.{profile.profile_name}.profile_config", fromlist=[profile.__class__.__name__])
        profile_class = getattr(profile_module, profile.__class__.__name__)
        
        # Check for profile-specific constants
        if hasattr(profile_class, 'CONSTANTS'):
            constants = profile_class.CONSTANTS
            
            # Override system config with profile constants
            if hasattr(constants, 'GENERATION_MODEL'):
                config.generation_model = constants.GENERATION_MODEL
            if hasattr(constants, 'EMBEDDING_MODEL'):
                config.embedding_model = constants.EMBEDDING_MODEL
            if hasattr(constants, 'TEMPERATURE'):
                config.temperature = constants.TEMPERATURE
            if hasattr(constants, 'MAX_TOKENS'):
                config.max_tokens = constants.MAX_TOKENS
            if hasattr(constants, 'VECTOR_STORE_TYPE'):
                config.vector_store_type = constants.VECTOR_STORE_TYPE
            if hasattr(constants, 'CHUNK_SIZE'):
                config.chunk_size = constants.CHUNK_SIZE
            if hasattr(constants, 'CHUNK_OVERLAP'):
                config.chunk_overlap = constants.CHUNK_OVERLAP
            if hasattr(constants, 'TOP_K'):
                config.top_k = constants.TOP_K
            if hasattr(constants, 'MAX_ITERATIONS'):
                config.max_iterations = constants.MAX_ITERATIONS
            if hasattr(constants, 'API_PORT'):
                config.api_port = constants.API_PORT
            if hasattr(constants, 'MCP_PORT'):
                config.mcp_port = constants.MCP_PORT
            if hasattr(constants, 'SAMPLE_SIZE'):
                config.sample_size = constants.SAMPLE_SIZE
            if hasattr(constants, 'LOG_LEVEL'):
                config.log_level = constants.LOG_LEVEL
            if hasattr(constants, 'LOG_TO_FILE'):
                config.log_to_file = constants.LOG_TO_FILE
            if hasattr(constants, 'LOG_TO_CONSOLE'):
                config.log_to_console = constants.LOG_TO_CONSOLE
            if hasattr(constants, 'LANGSMITH_API_KEY'):
                config.langsmith_api_key = constants.LANGSMITH_API_KEY
            if hasattr(constants, 'LANGSMITH_PROJECT'):
                config.langsmith_project = constants.LANGSMITH_PROJECT
            if hasattr(constants, 'ENABLE_TRACING'):
                config.enable_tracing = constants.ENABLE_TRACING
                
    except Exception as e:
        # If profile override fails, continue with default config
        pass
    
    return config


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
