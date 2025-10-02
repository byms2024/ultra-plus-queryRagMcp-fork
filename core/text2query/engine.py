#!/usr/bin/env python3
"""
Unified query synthesis engine with intelligent method selection and fallback mechanisms.
Combines traditional JSON-based synthesis with LangChain approaches.
"""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import time

from config.base_config import Config, load_profile
from config.logging_config import get_rag_logger
from config.profiles import DataProfile
from censor_utils.censoring import CensoringService
from config.providers.registry import LLMFactory

# Import synthesis methods
from .synthesis import QuerySynthesizer, LangChainQuerySynthesizer, LangChainAgentEngine
from .execution import QueryExecutor
from .data import DataManager
from .response import ResponseBuilder, StatsGenerator

logger = get_rag_logger()


class QuerySynthesisEngine:
    """
    Unified query synthesis engine with intelligent method selection.
    
    Supports multiple synthesis approaches:
    1. Traditional JSON-based synthesis
    2. LangChain direct pandas code generation  
    3. LangChain agent-based approach
    
    Provides intelligent selection, fallback mechanisms, and performance tracking.
    """
    
    def __init__(self, config: Config, profile: Optional[DataProfile] = None):
        self.config = config
        self.profile = profile or load_profile(config)
        
        # Initialize LLM provider (shared across all methods)
        self.llm_provider = self._create_llm_provider()
        logger.info(f"LLM provider initialized: {self.llm_provider}")
        # Initialize censoring with profile mappings
        self.censor = CensoringService()
        logger.info(f"Censoring service initialized: {self.censor}")
        # Initialize specialized modules
        self.data_manager = DataManager(config, self.profile)
        self.response_builder = ResponseBuilder(self.profile)
        self.stats_generator = StatsGenerator(self.profile)
        
        # Initialize synthesis engines
        self.traditional_synthesizer = None
        self.langchain_synthesizer = None
        self.langchain_agent = None
        
        # Performance tracking
        self.performance_stats = {
            "traditional": {"success_count": 0, "failure_count": 0, "avg_time": 0},
            "langchain_direct": {"success_count": 0, "failure_count": 0, "avg_time": 0},
            "langchain_agent": {"success_count": 0, "failure_count": 0, "avg_time": 0}
        }
        
        self._initialize_engines()
        
        # Load and process data
        self.df = self.data_manager.load_and_process_data()
    
    def _create_llm_provider(self):
        """Create LLM provider based on profile configuration."""
        provider_config = self.profile.get_provider_config()
        return LLMFactory.create(provider_config)
    
    def _initialize_engines(self):
        """Initialize all available query synthesis engines."""
        try:
            # Initialize traditional synthesizer
            self.traditional_synthesizer = QuerySynthesizer(self.config, self.profile)
            logger.info("Traditional synthesizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize traditional synthesizer: {e}")
        
        try:
            # Initialize LangChain synthesizer
            self.langchain_synthesizer = LangChainQuerySynthesizer(self.config, self.profile)
            logger.info("LangChain synthesizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain synthesizer: {e}")
        
        try:
            # Initialize LangChain agent
            self.langchain_agent = LangChainAgentEngine(self.config, self.profile)
            logger.info("LangChain agent initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain agent: {e}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available synthesis methods."""
        methods = []
        if self.traditional_synthesizer:
            methods.append("traditional")
        if self.langchain_synthesizer:
            methods.append("langchain_direct")
        if self.langchain_agent:
            methods.append("langchain_agent")
        return methods
    
    def _select_best_method(self, question: str, df: pd.DataFrame) -> str:
        """Intelligently select the best synthesis method based on context."""
        try:
            available_methods = self.get_available_methods()
            
            if not available_methods:
                raise RuntimeError("No synthesis methods available")
            
            # Simple heuristic-based selection
            # For complex queries with aggregations, prefer LangChain
            complex_indicators = ["group by", "aggregate", "sum", "average", "mean", "count", "max", "min"]
            if any(indicator in question.lower() for indicator in complex_indicators):
                if "langchain_direct" in available_methods:
                    return "langchain_direct"
                elif "langchain_agent" in available_methods:
                    return "langchain_agent"
            
            # For simple filtering queries, prefer traditional method
            if "traditional" in available_methods:
                return "traditional"
            
            # Default to first available method
            return available_methods[0]
            
        except Exception as e:
            logger.warning(f"Failed to select best method: {e}")
            return "traditional" if self.traditional_synthesizer else "langchain_direct"
    
    def _update_performance_stats(self, method: str, success: bool, execution_time: float):
        """Update performance statistics for method selection."""
        if method in self.performance_stats:
            if success:
                self.performance_stats[method]["success_count"] += 1
                # Update average time using a simple moving average
                current_avg = self.performance_stats[method]["avg_time"]
                success_count = self.performance_stats[method]["success_count"]
                self.performance_stats[method]["avg_time"] = (
                    (current_avg * (success_count - 1) + execution_time) / success_count
                ) if success_count > 0 else execution_time
            else:
                self.performance_stats[method]["failure_count"] += 1
    
    def synthesize_query(
        self,
        question: str,
        method: str = "auto",
        attempt_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesize a query using the specified method or auto-select the best method.
        
        Args:
            question: Natural language question
            method: Synthesis method ("auto", "traditional", "langchain_direct", "langchain_agent")
        
        Returns:
            Query specification or result dictionary
        """
        original_method = method
        attempt_history = attempt_history or []
        if method == "auto":
            method = self._select_best_method(question, self.df)
        
        logger.info(
            f"📋 Using synthesis method: {method}"
            + (f" (requested: {original_method})" if original_method != method else "")
        )
        
        try:
            start_time = time.time()
            logger.info(f"🔄 Starting synthesis with {method}...")
            result = self._synthesize_with_method(method, question)
            execution_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(method, True, execution_time)
            
            logger.info(f"✅ Synthesis completed with {method} in {execution_time:.2f}s")

            attempt_history.append(
                {
                    "method": method,
                    "success": True,
                    "duration": execution_time,
                }
            )
            
            # Add method metadata
            if result:
                result["synthesis_method"] = method
                result["execution_time"] = execution_time
            else:
                result = {
                    "synthesis_method": method,
                    "execution_time": execution_time,
                }

            result["attempt_history"] = attempt_history.copy()

            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Synthesis failed with method {method} after {execution_time:.2f}s: {e}")
            self._update_performance_stats(method, False, 0)

            attempt_history.append(
                {
                    "method": method,
                    "success": False,
                    "duration": execution_time,
                }
            )

            lost_time = sum(a["duration"] for a in attempt_history if not a["success"])
            logger.warning(f"⏱️ Time lost on {method}: {execution_time:.2f}s (cumulative lost: {lost_time:.2f}s)")
            
            # Try fallback methods
            if method != "traditional" and self.traditional_synthesizer:
                logger.info(
                    f"🔄 Trying traditional method as fallback (cumulative lost time: {lost_time:.2f}s)"
                )
                return self.synthesize_query(question, "traditional", attempt_history)
            elif method != "langchain_direct" and self.langchain_synthesizer:
                logger.info(
                    f"🔄 Trying LangChain direct method as fallback (cumulative lost time: {lost_time:.2f}s)"
                )
                return self.synthesize_query(question, "langchain_direct", attempt_history)
            
            return {
                "error": f"All synthesis methods failed. Last error: {e}",
                "query_type": "synthesis_error",
                "synthesis_method": method,
                "attempt_history": attempt_history.copy(),
            }
    
    def _synthesize_with_method(self, method: str, question: str) -> Optional[Dict[str, Any]]:
        """Synthesize query using the specified method."""
        first_rows_hint = self.data_manager.get_sample_data(3)
        
        if method == "traditional" and self.traditional_synthesizer:
            return self.traditional_synthesizer.synthesize(question, first_rows_hint)
        
        elif method == "langchain_direct" and self.langchain_synthesizer:
            return self.langchain_synthesizer.synthesize(question, self.df, first_rows_hint)
        
        elif method == "langchain_agent" and self.langchain_agent:
            return self.langchain_agent.synthesize_with_agent(question, self.df, first_rows_hint)
        
        else:
            raise ValueError(f"Synthesis method '{method}' not available")
    
    def execute_query(self, question: str, method: str = "auto") -> Dict[str, Any]:
        """
        Execute a complete query: synthesis + execution.
        
        Args:
            question: Natural language question
            method: Synthesis method ("auto", "traditional", "langchain_direct", "langchain_agent")
        
        Returns:
            Dictionary with result, sources, stats, and metadata
        """
        query_start = time.time()
        try:
            # Synthesize the query
            synthesis_start = time.time()
            logger.info(f"🔍 Starting query synthesis...")
            query_spec = self.synthesize_query(question, method)
            synthesis_time = time.time() - synthesis_start
            attempt_history = (
                query_spec.get("attempt_history", [])
                if isinstance(query_spec, dict)
                else []
            )
            
            if not query_spec or "error" in query_spec:
                logger.warning(f"⚠️ Synthesis returned error after {synthesis_time:.2f}s")
                response = {
                    "answer": f"Error: {query_spec.get('error', 'Synthesis failed') if query_spec else 'No result'}",
                    "sources": [],
                    "stats": {},
                    "method": query_spec.get("synthesis_method", method) if query_spec else method,
                    "execution_time": query_spec.get("execution_time", 0) if query_spec else 0,
                }
                if attempt_history:
                    total_attempt_time = sum(a["duration"] for a in attempt_history)
                    total_lost_time = sum(
                        a["duration"] for a in attempt_history if not a["success"]
                    )
                    logger.info(
                        "⏱️ Text2Query attempts exhausted | total_attempt_time=%.2fs | total_lost_time=%.2fs | attempts=%s",
                        total_attempt_time,
                        total_lost_time,
                        ", ".join(
                            f"{a['method']}:{'ok' if a['success'] else 'fail'}:{a['duration']:.2f}s"
                            for a in attempt_history
                        ),
                    )
                    response["attempt_history"] = attempt_history
                    response["total_attempt_time"] = total_attempt_time
                    response["total_lost_time"] = total_lost_time
                return response
            
            # Execute the query based on its type
            if query_spec.get("query_type") in ["langchain_direct", "langchain_series", "langchain_scalar", "langchain_agent"]:
                # LangChain methods return results directly
                logger.info(f"⚡ Building response from LangChain result...")
                response = self.response_builder.build_response(query_spec.get("result"), query_spec)
                total_time = time.time() - query_start
                logger.info(f"✅ Query completed in {total_time:.2f}s (synthesis: {synthesis_time:.2f}s)")
                payload = {
                    "answer": response.get("answer", ""),
                    "sources": response.get("sources", []),
                    "confidence": response.get("confidence", "medium"),
                    "stats": self._generate_stats_from_result(query_spec.get("result")),
                    "method": query_spec.get("synthesis_method", method),
                    "execution_time": query_spec.get("execution_time", 0),
                }
                visual_answer = response.get("visual_answer")
                if visual_answer:
                    payload["visual_answer"] = visual_answer
                if response.get("preview"):
                    payload["preview"] = response["preview"]
                if attempt_history:
                    total_attempt_time = sum(a["duration"] for a in attempt_history)
                    total_lost_time = sum(
                        a["duration"] for a in attempt_history if not a["success"]
                    )
                    payload["attempt_history"] = attempt_history
                    payload["total_attempt_time"] = total_attempt_time
                    payload["total_lost_time"] = total_lost_time
                    logger.info(
                        "⏱️ Text2Query attempts summary | total_attempt_time=%.2fs | total_lost_time=%.2fs",
                        total_attempt_time,
                        total_lost_time,
                    )
                return payload
            else:
                # Traditional method needs execution via QueryExecutor
                exec_start = time.time()
                logger.info(f"⚡ Executing traditional query...")
                executor = QueryExecutor(self.profile)
                df_result = executor.apply(self.df, query_spec)
                exec_time = time.time() - exec_start
                
                logger.info(f"⚡ Building response...")
                response = self.response_builder.build_response(df_result, query_spec)
                total_time = time.time() - query_start
                logger.info(f"✅ Query completed in {total_time:.2f}s (synthesis: {synthesis_time:.2f}s, execution: {exec_time:.2f}s)")
                payload = {
                    "answer": response.get("answer", ""),
                    "sources": response.get("sources", []),
                    "confidence": response.get("confidence", "medium"),
                    "stats": self._generate_stats_from_result(df_result),
                    "method": query_spec.get("synthesis_method", method),
                    "execution_time": query_spec.get("execution_time", 0),
                }
                visual_answer = response.get("visual_answer")
                if visual_answer:
                    payload["visual_answer"] = visual_answer
                if response.get("preview"):
                    payload["preview"] = response["preview"]
                if attempt_history:
                    total_attempt_time = sum(a["duration"] for a in attempt_history)
                    total_lost_time = sum(
                        a["duration"] for a in attempt_history if not a["success"]
                    )
                    payload["attempt_history"] = attempt_history
                    payload["total_attempt_time"] = total_attempt_time
                    payload["total_lost_time"] = total_lost_time
                    logger.info(
                        "⏱️ Text2Query attempts summary | total_attempt_time=%.2fs | total_lost_time=%.2fs",
                        total_attempt_time,
                        total_lost_time,
                    )
                return payload
                
        except Exception as e:
            total_time = time.time() - query_start
            logger.error(f"❌ Query execution failed after {total_time:.2f}s: {e}")
            return {
                "answer": f"Error: {e}",
                "sources": [],
                "stats": {},
                "method": method,
                "execution_time": 0
            }
    
    def _create_sources_from_result(self, result: Union[pd.DataFrame, pd.Series, Any]) -> List[Dict[str, Any]]:
        """Create source dictionaries from DataFrame/Series rows or scalar result."""
        if isinstance(result, pd.DataFrame):
            return self.profile.create_sources_from_df(result)
        elif isinstance(result, pd.Series):
            # Convert series to a DataFrame for source creation
            return self.profile.create_sources_from_df(result.to_frame(name="value"))
        elif pd.api.types.is_scalar(result):
            return [{"result": str(result)}]
        return []
    
    def _generate_stats_from_result(self, result: Union[pd.DataFrame, pd.Series, Any]) -> Dict[str, Any]:
        """Generate statistics from query result."""
        stats = {}
        if isinstance(result, pd.DataFrame):
            stats["rows"] = len(result)
            stats["columns"] = len(result.columns)
        elif isinstance(result, pd.Series):
            stats["rows"] = len(result)
            stats["columns"] = 1
        elif pd.api.types.is_scalar(result):
            stats["value"] = str(result)
        return stats
    
    def answer_question_with_langchain(self, question: str, method: str = "auto") -> Dict[str, Any]:
        """
        Answer a question using the unified engine with LangChain support.
        
        Args:
            question: Natural language question
            method: Synthesis method ("auto", "traditional", "langchain_direct", "langchain_agent")
        
        Returns:
            Dictionary with answer, sources, stats, and metadata
        """
        return self.execute_query(question, method)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all methods."""
        return self.performance_stats.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if self.df is None:
            return {}
        
        stats = {
            "total_rows": len(self.df),
            "total_records": len(self.df),  # Alias for backward compatibility
            "total_columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        # Add null counts
        null_counts = self.df.isnull().sum()
        stats["null_counts"] = null_counts.to_dict()
        
        return stats