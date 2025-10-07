#!/usr/bin/env python3
"""
Provider-agnostic factories for LLM and Embeddings instances.

Profiles provide a ProviderConfig; factories create the appropriate LangChain
objects without the rest of the code importing provider-specific classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Supported provider SDK imports (lazy used behind factories)
try:  # Google Generative AI
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore
    GoogleGenerativeAIEmbeddings = None  # type: ignore

# Vertex AI support removed to avoid ADC complexities


@dataclass
class ProviderConfig:
    provider: str = "google"  # e.g., google | openai | anthropic | cohere | azure_openai
    generation_model: str = ""
    embedding_model: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


class LLMFactory:
    @staticmethod
    def create(config: ProviderConfig):
        provider = (config.provider or "google").lower()
        if provider == "google":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain_google_genai is not available")
            api_key = config.credentials.get("api_key", "")
            temperature = config.extras.get("temperature", 0.2)
            max_tokens = config.extras.get("max_tokens", 2048)
            if not api_key:
                raise ValueError("Google provider requires 'api_key' in ProviderConfig.credentials")
            # Optional timeout/retry controls
            timeout_seconds = config.extras.get("timeout", None)
            max_retries = config.extras.get("max_retries", 0)
            return ChatGoogleGenerativeAI(
                model=config.generation_model,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
                # Fail fast on network/DNS errors
                max_retries=max_retries,
                # If provided, enforce per-request timeout
                **({"timeout": timeout_seconds} if timeout_seconds is not None else {}),
            )
        # Placeholder stubs for future providers
        elif provider == "openai":
            raise NotImplementedError("OpenAI provider not yet wired")
        elif provider == "anthropic":
            raise NotImplementedError("Anthropic provider not yet wired")
        elif provider == "cohere":
            raise NotImplementedError("Cohere provider not yet wired")
        elif provider == "azure_openai":
            raise NotImplementedError("Azure OpenAI provider not yet wired")
        # 'vertex_ai' removed
        else:
            raise ValueError(f"Unknown provider: {provider}")


class EmbeddingsFactory:
    @staticmethod
    def create(config: ProviderConfig):
        provider = (config.provider or "google").lower()
        if provider == "google":
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError("langchain_google_genai is not available")
            api_key = config.credentials.get("api_key", "")
            if not api_key:
                raise ValueError("Google embeddings require 'api_key' in ProviderConfig.credentials")
            return GoogleGenerativeAIEmbeddings(
                model=config.embedding_model,
                google_api_key=api_key,
            )
        elif provider == "openai":
            raise NotImplementedError("OpenAI embeddings not yet wired")
        elif provider == "cohere":
            raise NotImplementedError("Cohere embeddings not yet wired")
        elif provider == "azure_openai":
            raise NotImplementedError("Azure OpenAI embeddings not yet wired")
        else:
            raise ValueError(f"Unknown provider: {provider}")


