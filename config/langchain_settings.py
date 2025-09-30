"""
Backward compatibility wrapper for LangChain configuration.
This module provides the old interface while using the new profile-based system.
"""

from dataclasses import dataclass
from typing import Optional

# Import the new configuration system
from .base_config import (
    load_system_config, 
    get_profile, 
    get_data_file_path,
    get_vector_store_path,
    get_api_port,
    get_mcp_port,
    get_google_api_key,
    get_generation_model,
    get_embedding_model,
    get_temperature,
    get_max_tokens,
    get_chunk_size,
    get_chunk_overlap,
    get_top_k,
    get_max_iterations,
    get_sample_size,
    get_langsmith_api_key,
    get_langsmith_project,
    get_enable_tracing
)


@dataclass
class LangChainConfig:
    """
    Backward compatibility class for LangChain configuration.
    This class provides the same interface as before but uses the new profile system.
    """
    
    def __init__(self):
        """Initialize configuration using the new profile-based system."""
        # Get the current profile and system config
        self._profile = get_profile()
        self._config = load_system_config()
        
        # Map to the old interface
        self.generation_model = get_generation_model()
        self.embedding_model = get_embedding_model()
        self.temperature = get_temperature()
        self.max_tokens = get_max_tokens()
        self.vector_store_type = self._config.vector_store_type
        self.vector_store_path = get_vector_store_path()
        self.chunk_size = get_chunk_size()
        self.chunk_overlap = get_chunk_overlap()
        self.top_k = get_top_k()
        self.max_iterations = get_max_iterations()
        self.mcp_port = get_mcp_port()
        self.api_port = get_api_port()
        self.csv_file = get_data_file_path()  # Now uses profile-based path
        self.sample_size = get_sample_size()
        self.google_api_key = get_google_api_key()
        self.langsmith_api_key = get_langsmith_api_key()
        self.langsmith_project = get_langsmith_project()
        self.enable_tracing = get_enable_tracing()
    
    @property
    def profile(self):
        """Get the current profile."""
        return self._profile
    
    def get_profile(self):
        """Get the current profile (convenience method)."""
        return self._profile


def load_langchain_config() -> LangChainConfig:
    """
    Load LangChain configuration using the new profile-based system.
    This function maintains backward compatibility.
    """
    return LangChainConfig()
