"""
Strategy Factory Implementation.

This module implements the Factory Pattern for creating RAG retrieval strategies.
It provides a centralized way to create and manage different retrieval strategies
based on configuration.

Factory Pattern Benefits:
- Centralized strategy creation
- Easy to add new strategies
- Runtime strategy selection
- Consistent strategy interface
- Error handling and validation
"""

from typing import Dict, Type, Optional
from .base_strategy import BaseRetrievalStrategy
from .top_k_strategy import TopKStrategy
from .hybrid_strategy import HybridStrategy
from core.rag.config.rag_config import RAGConfig


class StrategyFactory:
    """
    Factory class for creating RAG retrieval strategies.
    
    This factory implements the Factory Pattern to provide a centralized way
    to create and manage different retrieval strategies. It supports:
    - Strategy registration and discovery
    - Runtime strategy creation based on configuration
    - Strategy validation and error handling
    - Extensibility for custom strategies
    
    Factory Pattern Implementation:
    - Creator: StrategyFactory (creates strategy objects)
    - Product: BaseRetrievalStrategy (strategy interface)
    - Concrete Product: TopKStrategy, HybridStrategy (specific strategies)
    
    Example:
        >>> factory = StrategyFactory()
        >>> config = RAGConfig(retrieval_strategy="hybrid")
        >>> strategy = factory.create_strategy(config)
        >>> results = strategy.search_relevant_chunks(query, vectorstore, config)
    """
    
    # Registry of available strategies
    _strategies: Dict[str, Type[BaseRetrievalStrategy]] = {
        "top_k": TopKStrategy,
        "hybrid": HybridStrategy,
    }
    
    @classmethod
    def create_strategy(cls, config: RAGConfig) -> BaseRetrievalStrategy:
        """
        Create a retrieval strategy based on configuration.
        
        This method is the main entry point for strategy creation. It:
        1. Validates the configuration
        2. Looks up the appropriate strategy class
        3. Creates and returns a strategy instance
        4. Handles errors gracefully
        
        Args:
            config: RAG configuration containing strategy selection
            
        Returns:
            Strategy instance implementing BaseRetrievalStrategy
            
        Raises:
            ValueError: If strategy name is invalid or configuration is invalid
            RuntimeError: If strategy creation fails
            
        Example:
            >>> factory = StrategyFactory()
            >>> config = RAGConfig(retrieval_strategy="top_k", top_k=50)
            >>> strategy = factory.create_strategy(config)
            >>> print(f"Created strategy: {strategy.get_strategy_name()}")
            "top_k"
            
            >>> config = RAGConfig(retrieval_strategy="hybrid", similarity_threshold=0.8)
            >>> strategy = factory.create_strategy(config)
            >>> print(f"Created strategy: {strategy.get_strategy_description()}")
            "Hybrid strategy: Quality-first with intelligent fallback mechanisms"
        """
        # Validate configuration first
        config.validate()
        
        # Get strategy name from config
        strategy_name = config.retrieval_strategy
        
        # Check if strategy is registered
        if strategy_name not in cls._strategies:
            available_strategies = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown retrieval strategy: '{strategy_name}'. "
                f"Available strategies: {available_strategies}"
            )
        
        try:
            # Create strategy instance
            strategy_class = cls._strategies[strategy_name]
            strategy = strategy_class()
            
            # Validate strategy-specific configuration
            strategy.validate_config(config)
            
            return strategy
            
        except Exception as e:
            raise RuntimeError(f"Failed to create strategy '{strategy_name}': {str(e)}")
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseRetrievalStrategy]) -> None:
        """
        Register a new retrieval strategy.
        
        This method allows extending the factory with custom strategies.
        It validates that the strategy class implements the required interface.
        
        Args:
            name: Strategy name (must be unique)
            strategy_class: Strategy class implementing BaseRetrievalStrategy
            
        Raises:
            ValueError: If name is already registered or class is invalid
            TypeError: If strategy_class doesn't inherit from BaseRetrievalStrategy
            
        Example:
            >>> class CustomStrategy(BaseRetrievalStrategy):
            ...     def search_relevant_chunks(self, query, vectorstore, config):
            ...         return []
            ...     def get_strategy_name(self):
            ...         return "custom"
            ...     def get_strategy_description(self):
            ...         return "Custom strategy"
            
            >>> StrategyFactory.register_strategy("custom", CustomStrategy)
            >>> config = RAGConfig(retrieval_strategy="custom")
            >>> strategy = StrategyFactory.create_strategy(config)
            >>> print(strategy.get_strategy_name())
            "custom"
        """
        # Validate strategy class
        if not issubclass(strategy_class, BaseRetrievalStrategy):
            raise TypeError(
                f"Strategy class must inherit from BaseRetrievalStrategy, "
                f"got {strategy_class.__name__}"
            )
        
        # Check for duplicate registration
        if name in cls._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")
        
        # Register the strategy
        cls._strategies[name] = strategy_class
    
    @classmethod
    def unregister_strategy(cls, name: str) -> None:
        """
        Unregister a retrieval strategy.
        
        This method removes a strategy from the factory registry.
        Built-in strategies cannot be unregistered.
        
        Args:
            name: Strategy name to unregister
            
        Raises:
            ValueError: If strategy is not registered or is a built-in strategy
            
        Example:
            >>> StrategyFactory.unregister_strategy("custom")
            >>> # Built-in strategies cannot be unregistered
            >>> StrategyFactory.unregister_strategy("top_k")  # Raises ValueError
        """
        if name not in cls._strategies:
            raise ValueError(f"Strategy '{name}' is not registered")
        
        # Prevent unregistering built-in strategies
        built_in_strategies = {"top_k", "hybrid"}
        if name in built_in_strategies:
            raise ValueError(f"Cannot unregister built-in strategy '{name}'")
        
        del cls._strategies[name]
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, str]:
        """
        List all available strategies with their descriptions.
        
        Returns:
            Dictionary mapping strategy names to their descriptions
            
        Example:
            >>> strategies = StrategyFactory.list_available_strategies()
            >>> for name, description in strategies.items():
            ...     print(f"{name}: {description}")
            top_k: Top-K strategy: Returns exactly K results based on similarity, no quality filtering
            hybrid: Hybrid strategy: Quality-first with intelligent fallback mechanisms
        """
        strategies_info = {}
        for name, strategy_class in cls._strategies.items():
            # Create a temporary instance to get description
            try:
                temp_strategy = strategy_class()
                strategies_info[name] = temp_strategy.get_strategy_description()
            except Exception:
                strategies_info[name] = f"Strategy class: {strategy_class.__name__}"
        
        return strategies_info
    
    @classmethod
    def get_strategy_info(cls, name: str) -> Dict[str, any]:
        """
        Get detailed information about a specific strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Dictionary containing strategy information:
            - name: Strategy name
            - description: Strategy description
            - performance_characteristics: Performance metrics
            - class_name: Strategy class name
            
        Raises:
            ValueError: If strategy is not registered
            
        Example:
            >>> info = StrategyFactory.get_strategy_info("hybrid")
            >>> print(f"Strategy: {info['name']}")
            >>> print(f"Description: {info['description']}")
            >>> print(f"Performance: {info['performance_characteristics']}")
        """
        if name not in cls._strategies:
            available_strategies = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{name}'. Available strategies: {available_strategies}"
            )
        
        strategy_class = cls._strategies[name]
        temp_strategy = strategy_class()
        
        return {
            "name": temp_strategy.get_strategy_name(),
            "description": temp_strategy.get_strategy_description(),
            "performance_characteristics": temp_strategy.get_performance_characteristics(),
            "class_name": strategy_class.__name__
        }
    
    @classmethod
    def validate_strategy_config(cls, config: RAGConfig) -> Dict[str, any]:
        """
        Validate configuration against all available strategies.
        
        This method checks if the configuration is valid for any strategy
        and provides detailed validation results.
        
        Args:
            config: RAG configuration to validate
            
        Returns:
            Dictionary containing validation results:
            - valid: Whether configuration is valid
            - strategy_name: Strategy name from config
            - errors: List of validation errors
            - warnings: List of validation warnings
            
        Example:
            >>> config = RAGConfig(retrieval_strategy="hybrid", similarity_threshold=0.8)
            >>> validation = StrategyFactory.validate_strategy_config(config)
            >>> if validation['valid']:
            ...     print(f"Configuration valid for {validation['strategy_name']}")
            ... else:
            ...     print(f"Validation errors: {validation['errors']}")
        """
        validation_result = {
            "valid": False,
            "strategy_name": config.retrieval_strategy,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Basic config validation
            config.validate()
            
            # Strategy-specific validation
            if config.retrieval_strategy in cls._strategies:
                strategy_class = cls._strategies[config.retrieval_strategy]
                temp_strategy = strategy_class()
                temp_strategy.validate_config(config)
                validation_result["valid"] = True
            else:
                validation_result["errors"].append(
                    f"Unknown strategy: {config.retrieval_strategy}"
                )
                
        except Exception as e:
            validation_result["errors"].append(str(e))
        
        return validation_result
