#!/usr/bin/env python3
"""
LangChain agent-based query engine that uses PythonREPL for safe code execution.
Integrates with existing profile system and enhanced data cleaning.
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from config.logging_config import get_logger
from config.base_config import Config
from config.profiles import DataProfile
from ..data import build_schema_description, validate_dataframe_for_langchain
from config.providers.registry import LLMFactory

logger = get_logger(__name__)


class LangChainAgentEngine:
    """
    LangChain agent-based query engine using PythonREPL for safe execution.
    Uses the agent approach from the sample code but integrates with our profile system.
    """
    
    def __init__(self, config: Config, profile: DataProfile):
        self.config = config
        self.profile = profile
        
        # Create LangChain provider using the profile's provider configuration
        provider_config = profile.get_provider_config()
        
        # Enable LangChain mode with agent
        provider_config.use_langchain = True
        provider_config.langchain_provider = "openai"  # Default to OpenAI for LangChain
        provider_config.use_agent = True
        provider_config.python_repl_enabled = True
        provider_config.verbose = False  # Set to True for debugging
        
        self.llm_provider = LLMFactory.create(provider_config)
        self.allowed_columns = profile.required_columns
        
        logger.info(f"Initialized LangChain agent engine with provider: {provider_config.provider}")
    
    def build_agent(self, df: pd.DataFrame) -> Any:
        """
        Build a LangChain agent with PythonREPL tool for the given DataFrame.
        This is based on the sample code's build_agent function.
        """
        try:
            # Check if the provider supports agents
            if not hasattr(self.llm_provider, 'is_agent_available') or not self.llm_provider.is_agent_available():
                raise RuntimeError("LangChain agent is not available with current provider configuration")
            
            # The agent is already created in the LangChainLLMWrapper
            agent = self.llm_provider.agent
            
            if not agent:
                raise RuntimeError("Failed to create LangChain agent")
            
            logger.info("LangChain agent created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to build LangChain agent: {e}")
            raise
    
    def run_agent_query(self, query: str, df: pd.DataFrame) -> Union[str, Dict[str, Any]]:
        """
        Run a query using the LangChain agent with PythonREPL.
        This integrates the sample code's agent approach with our profile system.
        """
        try:
            # Validate DataFrame for LangChain processing
            validation = validate_dataframe_for_langchain(df, self.profile)
            if not validation['is_valid']:
                logger.warning(f"DataFrame validation failed: {validation['errors']}")
            
            # Build the agent
            agent = self.build_agent(df)
            
            # Build context-aware query
            context_query = self._build_context_query(query, df)
            
            # Run the agent query
            logger.info(f"Running agent query: {query}")
            result = agent.run(context_query)
            
            logger.info(f"Agent query completed successfully")
            return str(result)
            
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            return f"Agent execution error: {e}"
    
    def _build_context_query(self, query: str, df: pd.DataFrame) -> str:
        """Build a context-aware query for the agent."""
        try:
            # Build schema description
            schema_description = build_schema_description(df, self.profile)
            
            # Get profile-specific context
            profile_context = self._get_profile_context()
            
            # Build the complete context query
            context_parts = [
                f"You have access to a pandas DataFrame called 'df' with the following schema:",
                "",
                schema_description,
                "",
                f"Profile context: {profile_context}",
                "",
                f"User question: {query}",
                "",
                "Use the PythonREPL to write and execute pandas code to answer the question.",
                "Make sure to assign your final result to a variable so it can be returned."
            ]
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to build context query: {e}")
            return f"Using DataFrame 'df', answer this question: {query}"
    
    def _get_profile_context(self) -> str:
        """Get profile-specific context for the agent."""
        try:
            context_parts = []
            
            # Add domain terminology
            if hasattr(self.profile, 'get_domain_terminology'):
                terminology = self.profile.get_domain_terminology()
                if terminology:
                    context_parts.append(f"Domain terminology: {terminology}")
            
            # Add language information
            if hasattr(self.profile, 'get_language'):
                language = self.profile.get_language()
                context_parts.append(f"Language: {language}")
            
            # Add example queries
            if hasattr(self.profile, 'get_example_queries'):
                examples = self.profile.get_example_queries()
                if examples:
                    context_parts.append(f"Example queries: {', '.join(examples[:3])}")  # Limit to 3 examples
            
            return "; ".join(context_parts) if context_parts else "Standard data analysis profile"
            
        except Exception as e:
            logger.warning(f"Failed to get profile context: {e}")
            return "Standard data analysis profile"
    
    def synthesize_with_agent(self, question: str, df: pd.DataFrame, 
                            df_first_rows_hint: str = "") -> Optional[Dict[str, Any]]:
        """
        Main synthesis method using LangChain agent.
        Returns a dictionary compatible with the existing query executor.
        """
        try:
            # Run the agent query
            result = self.run_agent_query(question, df)
            
            if isinstance(result, str) and "error" in result.lower():
                # Return error in expected format
                return {
                    "error": result,
                    "query_type": "agent_error"
                }
            
            # Convert result to expected format
            return self._format_agent_result(result, question)
            
        except Exception as e:
            logger.error(f"LangChain agent synthesis failed: {e}")
            return {
                "error": f"LangChain agent synthesis failed: {e}",
                "query_type": "agent_error"
            }
    
    def _format_agent_result(self, result: str, question: str) -> Dict[str, Any]:
        """Format the agent result for compatibility with existing query executor."""
        try:
            # The agent result is typically a string description
            # We'll create a simple DataFrame representation
            result_df = pd.DataFrame([{
                "question": question,
                "answer": result,
                "method": "langchain_agent"
            }])
            
            return {
                "filters": [],  # Agent generates code directly, no explicit filters
                "aggregations": [],
                "sort_by": [],
                "limit": 1,
                "query_type": "langchain_agent",
                "result": result_df,
                "agent_generated": True,
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to format agent result: {e}")
            return {
                "error": f"Failed to format agent result: {e}",
                "query_type": "agent_error"
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the LangChain agent configuration."""
        try:
            info = {
                "agent_available": False,
                "provider": self.llm_provider.provider if hasattr(self.llm_provider, 'provider') else 'unknown',
                "model": self.llm_provider.model if hasattr(self.llm_provider, 'model') else 'unknown',
                "tools_available": []
            }
            
            if hasattr(self.llm_provider, 'is_agent_available'):
                info["agent_available"] = self.llm_provider.is_agent_available()
            
            if hasattr(self.llm_provider, 'get_available_tools'):
                info["tools_available"] = self.llm_provider.get_available_tools()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get agent info: {e}")
            return {
                "agent_available": False,
                "error": str(e)
            }
