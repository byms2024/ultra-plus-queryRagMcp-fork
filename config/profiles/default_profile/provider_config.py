#!/usr/bin/env python3
"""
Provider configuration for the default profile.
"""

from config.providers.registry import ProviderConfig


def get_provider_config() -> ProviderConfig:
    from .config_api_keys import GCP_API_KEY
    return ProviderConfig(
        provider="google",
        # generation_model="gemini-1.5-flash", # no longer supported
        generation_model="gemini-2.5-flash",
        embedding_model="text-embedding-004",
        credentials={"api_key": GCP_API_KEY},
        extras={"temperature": 0.2, "max_tokens": 2048},
    )


