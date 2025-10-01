"""
RAG Configuration Module.

This module defines the RAGConfig dataclass that encapsulates all RAG-specific configuration
parameters. It provides validation, defaults, and helper methods for RAG strategies.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """
    Configuration class for RAG (Retrieval-Augmented Generation) system.
    
    This class encapsulates all configuration parameters needed for RAG operations,
    including strategy selection, similarity thresholds, and result limits.
    
    Attributes:
        retrieval_strategy: Strategy to use for retrieval ("top_k" or "hybrid")
        top_k: Number of results for top_k strategy (1-1000)
        similarity_threshold: Minimum similarity score for quality filtering (0.0-1.0)
        max_search_with_threshold: Max candidates to search before filtering (10-1000)
        min_results_with_threshold: Minimum results to return as safety net (1-50)
        max_iterations: Maximum iterations for RAG operations (legacy)
    
    Example:
        >>> config = RAGConfig(
        ...     retrieval_strategy="hybrid",
        ...     similarity_threshold=0.7,
        ...     max_search_with_threshold=100
        ... )
        >>> print(config.get_hybrid_min_threshold())
        0.49
    """
    
    # Strategy Configuration
    retrieval_strategy: str = "hybrid"
    top_k: int = 50
    max_iterations: int = 10
    
    # Quality and Quantity Control
    similarity_threshold: float = 0.7
    max_search_with_threshold: int = 100
    min_results_with_threshold: int = 1
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is out of valid range.
            
        Example:
            >>> config = RAGConfig(similarity_threshold=1.5)
            >>> config.validate()  # Raises ValueError
        """
        if self.retrieval_strategy not in ["top_k", "hybrid"]:
            raise ValueError(f"Invalid retrieval_strategy: {self.retrieval_strategy}. Must be 'top_k' or 'hybrid'")
        
        if not (1 <= self.top_k <= 1000):
            raise ValueError(f"top_k must be between 1 and 1000, got {self.top_k}")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
        
        if not (10 <= self.max_search_with_threshold <= 1000):
            raise ValueError(f"max_search_with_threshold must be between 10 and 1000, got {self.max_search_with_threshold}")
        
        if not (1 <= self.min_results_with_threshold <= 50):
            raise ValueError(f"min_results_with_threshold must be between 1 and 50, got {self.min_results_with_threshold}")
        
        if not (1 <= self.max_iterations <= 100):
            raise ValueError(f"max_iterations must be between 1 and 100, got {self.max_iterations}")
    
    def get_hybrid_min_threshold(self) -> float:
        """
        Get the fallback similarity threshold for hybrid strategy.
        
        This is used when insufficient high-quality results are found.
        The fallback threshold is derived from the main threshold to ensure
        reasonable quality even in fallback scenarios.
        
        Returns:
            Fallback similarity threshold (typically 70% of main threshold)
            
        Example:
            >>> config = RAGConfig(similarity_threshold=0.7)
            >>> print(config.get_hybrid_min_threshold())
            0.49
        """
        return self.similarity_threshold * 0.7
    
    def is_top_k_strategy(self) -> bool:
        """
        Check if using top_k strategy.
        
        Returns:
            True if strategy is "top_k", False otherwise
            
        Example:
            >>> config = RAGConfig(retrieval_strategy="top_k")
            >>> print(config.is_top_k_strategy())
            True
        """
        return self.retrieval_strategy == "top_k"
    
    def is_hybrid_strategy(self) -> bool:
        """
        Check if using hybrid strategy.
        
        Returns:
            True if strategy is "hybrid", False otherwise
            
        Example:
            >>> config = RAGConfig(retrieval_strategy="hybrid")
            >>> print(config.is_hybrid_strategy())
            True
        """
        return self.retrieval_strategy == "hybrid"
    
    def get_effective_search_limit(self) -> int:
        """
        Get the effective search limit based on strategy.
        
        For top_k strategy, returns top_k.
        For hybrid strategy, returns max_search_with_threshold.
        
        Returns:
            Effective search limit for the current strategy
            
        Example:
            >>> config = RAGConfig(retrieval_strategy="top_k", top_k=50)
            >>> print(config.get_effective_search_limit())
            50
            >>> config = RAGConfig(retrieval_strategy="hybrid", max_search_with_threshold=100)
            >>> print(config.get_effective_search_limit())
            100
        """
        if self.is_top_k_strategy():
            return self.top_k
        else:  # hybrid strategy
            return self.max_search_with_threshold
    
    def get_strategy_description(self) -> str:
        """
        Get a human-readable description of the current strategy configuration.
        
        Returns:
            String description of the strategy and its parameters
            
        Example:
            >>> config = RAGConfig(retrieval_strategy="hybrid", similarity_threshold=0.8)
            >>> print(config.get_strategy_description())
            "Hybrid strategy: Search up to 100 candidates, filter by similarity >= 0.8, fallback to top 1 if insufficient"
        """
        if self.is_top_k_strategy():
            return f"Top-K strategy: Return exactly {self.top_k} results, no quality filtering"
        else:
            return (f"Hybrid strategy: Search up to {self.max_search_with_threshold} candidates, "
                   f"filter by similarity >= {self.similarity_threshold}, "
                   f"fallback to top {self.min_results_with_threshold} if insufficient")
    
    @classmethod
    def from_system_config(cls, system_config) -> 'RAGConfig':
        """
        Create RAGConfig from system configuration object.
        
        This method extracts RAG-specific configuration from a system configuration
        object and creates a RAGConfig instance.
        
        Args:
            system_config: System configuration object with RAG parameters
            
        Returns:
            RAGConfig instance with extracted parameters
            
        Example:
            >>> from config.base_config import load_system_config
            >>> system_config = load_system_config()
            >>> rag_config = RAGConfig.from_system_config(system_config)
            >>> print(rag_config.retrieval_strategy)
            "hybrid"
        """
        return cls(
            retrieval_strategy=getattr(system_config, 'retrieval_strategy', 'hybrid'),
            top_k=getattr(system_config, 'top_k', 50),
            max_iterations=getattr(system_config, 'max_iterations', 10),
            similarity_threshold=getattr(system_config, 'similarity_threshold', 0.7),
            max_search_with_threshold=getattr(system_config, 'max_search_with_threshold', 100),
            min_results_with_threshold=getattr(system_config, 'min_results_with_threshold', 1)
        )
    
    def __post_init__(self):
        """
        Post-initialization validation.
        
        Automatically validates configuration after dataclass initialization.
        """
        self.validate()
