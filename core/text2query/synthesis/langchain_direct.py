#!/usr/bin/env python3
"""
LangChain-based query synthesizer that generates pandas code directly.
Integrates with existing profile system and enhanced data cleaning.
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from config.logging_config import get_logger
from config.base_config import Config, load_system_config
from config.profiles import DataProfile
from ..utils.time_utils import parse_relative_date_range, StepTimer
from ..data import build_schema_description, validate_dataframe_for_langchain
from config.providers.registry import LLMFactory

logger = get_logger(__name__)


class LangChainQuerySynthesizer:
    """
    LangChain-based query synthesizer that generates pandas code directly.
    Uses the approach from the sample code but integrates with our profile system.
    """
    
    def __init__(self, config: Config, profile: DataProfile):
        self.config = config
        self.profile = profile
        self.strategy_name = "langchain_direct"
        
        # Create LangChain provider using the profile's provider configuration
        provider_config = profile.get_provider_config()
        
        # Enable LangChain mode
        provider_config.use_langchain = True
        # provider_config.langchain_provider = "openai"  # Default to OpenAI for LangChain
        
        self.llm_provider = LLMFactory.create(provider_config)
        self.allowed_columns = profile.required_columns
        
        logger.info(f"Initialized LangChain query synthesizer with provider: {provider_config.provider}")
    
    def llm_to_pandas(self, query: str, df: pd.DataFrame) -> Union[pd.DataFrame, str]:
        """
        Takes a natural language query, asks the LLM to generate pandas code,
        executes it safely, and returns the result.
        
        This is the core function from the sample code, enhanced with our profile system.
        """
        try:
            # Step 1: Build schema context using our enhanced system
            with StepTimer(logger, f"[{self.strategy_name}] Build schema description", "🧬"):
                schema_description = build_schema_description(df, self.profile)
            
            # Step 2: Get profile-specific system prompt and schema hints
            with StepTimer(logger, f"[{self.strategy_name}] Build system prompt", "🗒️"):
                system_prompt = self._build_system_prompt()
            with StepTimer(logger, f"[{self.strategy_name}] Build schema hints", "🔎"):
                schema_hints = self._build_schema_hints(schema_description)
            
            # Step 3: Handle date range parsing (integrate existing time utils)
            with StepTimer(logger, f"[{self.strategy_name}] Parse date context", "📆"):
                date_context = self._handle_date_context(query)
            
            # Step 4: Build the complete prompt
            with StepTimer(logger, f"[{self.strategy_name}] Build complete prompt", "🧩"):
                full_prompt = self._build_complete_prompt(
                    query, schema_description, system_prompt, schema_hints, date_context
                )
            
            # Step 5: Generate pandas code using LangChain
            with StepTimer(logger, f"[{self.strategy_name}] LLM call (LangChain)", "🤖"):
                # Use a minimal timeout wrapper; unified engine also has a budget timeout.
                response = self.llm_provider.invoke(full_prompt)
            
            with StepTimer(logger, f"[{self.strategy_name}] Extract generated code", "📝"):
                code = self._extract_code_from_response(response)
            
            logger.debug(f"Generated pandas code: {code}")
            
            # Step 6: Execute the code safely
            with StepTimer(logger, f"[{self.strategy_name}] Execute pandas code", "⚙️"):
                result = self._execute_pandas_code(code, df)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LangChain pandas generation failed: {e}")
            return f"Generation error: {e}"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt using profile-specific configuration."""
        try:
            # Get profile-specific system prompt
            base_prompt = self.profile.get_llm_system_prompt()
            
            # Add LangChain-specific instructions
            langchain_instructions = """
You are a Python data assistant specialized in generating pandas code.
Your task is to translate user questions into executable pandas code.

IMPORTANT INSTRUCTIONS:
1. Generate ONLY pandas code, no explanations
2. The code must assign the final result to a variable named 'result'
3. Use the provided DataFrame 'df' as the data source
4. Handle missing data appropriately with pandas methods
5. Use proper pandas syntax and best practices
6. If filtering by dates, use proper datetime comparison
7. Return the result as a pandas DataFrame or Series

MULTI-CRITERIA QUERY HANDLING:
When a query has multiple criteria (e.g., "recent ROs with lowest scores"), handle them in this order:
1. FIRST: Apply temporal filters (recent, latest, last month, etc.)
2. THEN: Apply ranking/sorting (lowest, highest, top, bottom)
3. FINALLY: Apply limits if needed

For queries like "recent ROs with lowest scores":
- Filter by date first: df[df['CREATE_DATE'] >= recent_cutoff]
- Then sort by score: .sort_values('SCORE', ascending=True)
- Consider limiting results: .head(10) for top results

EXAMPLE OUTPUT FORMAT:
```python
# Your pandas code here
result = df[df['column'] > value].groupby('group_column').agg({'metric': 'sum'})
```
"""
            
            return f"{base_prompt}\n\n{langchain_instructions}"
            
        except Exception as e:
            logger.warning(f"Failed to build system prompt: {e}")
            return "You are a Python data assistant. Generate pandas code to answer the user's question."
    
    def _build_schema_hints(self, schema_description: str) -> str:
        """Build schema hints using profile-specific information."""
        try:
            # Get profile-specific schema hints
            base_hints = self.profile.get_schema_hints(schema_description)
            
            # Add LangChain-specific schema information
            langchain_hints = f"""
SCHEMA INFORMATION:
{schema_description}

COLUMN USAGE GUIDELINES:
- Use only the columns listed above
- Pay attention to data types when writing filters
- Date columns should be compared using proper datetime objects
- Numeric columns can be used in aggregations and comparisons
- Text columns can be used in string operations and filtering
"""
            
            return f"{base_hints}\n\n{langchain_hints}"
            
        except Exception as e:
            logger.warning(f"Failed to build schema hints: {e}")
            return f"Available columns: {', '.join(self.allowed_columns)}"
    
    def _handle_date_context(self, query: str) -> str:
        """Handle date context using existing time utils."""
        try:
            window = parse_relative_date_range(query)
            if window is not None:
                start_date, end_date = window
                return f"""
DATE CONTEXT:
The query contains relative date references. Use these date ranges:
- Start Date: {start_date.date()}
- End Date: {end_date.date()}

When filtering by dates, use: df['CREATE_DATE'] >= '{start_date.date()}' and df['CREATE_DATE'] <= '{end_date.date()}'
"""
            
            # Handle common temporal terms that might not be parsed by time utils
            q_lower = query.lower()
            if any(term in q_lower for term in ['recent', 'latest', 'last']):
                return f"""
DATE CONTEXT:
The query contains temporal references ('recent', 'latest', 'last').
For 'recent' filter to last 7 days unless specified otherwise:
- Use: df['CREATE_DATE'] >= (df['CREATE_DATE'].max() - pd.Timedelta(days=7))
- Or: df['CREATE_DATE'] >= pd.Timestamp.now() - pd.Timedelta(days=7)
"""
            
            return ""
        except Exception as e:
            logger.warning(f"Failed to handle date context: {e}")
            return ""
    
    def _build_complete_prompt(self, query: str, schema_description: str, 
                             system_prompt: str, schema_hints: str, date_context: str) -> str:
        """Build the complete prompt for LangChain."""
        # Add specific guidance for multi-criteria queries
        multi_criteria_hint = ""
        q_lower = query.lower()
        if any(temporal in q_lower for temporal in ['recent', 'latest', 'last']) and \
           any(ranking in q_lower for ranking in ['lowest', 'highest', 'top', 'bottom', 'best', 'worst']):
            multi_criteria_hint = """
MULTI-CRITERIA QUERY DETECTED:
This query has both temporal and ranking criteria. Follow this sequence:
1. Filter by date/time first (recent, latest, etc.)
2. Then sort by the ranking criteria (lowest, highest, etc.)
3. Limit results if appropriate (head/tail)
"""
        
        prompt_parts = [
            system_prompt,
            "",
            schema_hints,
            "",
            date_context,
            "",
            multi_criteria_hint,
            "",
            f"USER QUESTION: {query}",
            "",
            "Generate pandas code to answer this question. The code must assign the result to a variable named 'result'."
        ]
        
        return "\n".join(filter(None, prompt_parts))
    
    def _extract_code_from_response(self, response) -> str:
        """Extract pandas code from LLM response."""
        try:
            # Handle LangChain response format
            if hasattr(response, 'content'):
                code = response.content.strip()
            elif isinstance(response, str):
                code = response.strip()
            else:
                code = str(response).strip()
            
            # Remove markdown code blocks if present
            if code.startswith("```") and code.endswith("```"):
                lines = code.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                code = '\n'.join(lines)
            
            # Remove any language specification
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```py"):
                code = code[5:]
            
            code = code.strip()
            # Harden: strip imports and dangerous builtins
            filtered_lines = []
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    continue
                if '__import__' in stripped or 'eval(' in stripped or 'exec(' in stripped:
                    continue
                filtered_lines.append(line)
            code = '\n'.join(filtered_lines).strip()
            
            # Ensure the code assigns to 'result'
            if 'result =' not in code:
                raise ValueError("Generated code must assign the final output to a variable named 'result'.")
            
            return code
            
        except Exception as e:
            logger.error(f"Failed to extract code from response: {e}")
            raise
    
    def _execute_pandas_code(self, code: str, df: pd.DataFrame) -> Union[pd.DataFrame, str]:
        """Execute pandas code safely in a restricted environment."""
        try:
            # Create a safe execution environment; use same dict for globals/locals so functions can access 'pd', etc.
            env = {
                'df': df,
                'pd': pd,
                'np': np,
                'datetime': datetime,
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round
                }
            }
            
            # Execute the code
            exec(code, env, env)
            
            # Try to return the 'result' variable
            if 'result' in env:
                result = env['result']
                
                # Validate the result
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    return result
                elif hasattr(result, '__len__'):
                    # Convert other iterables to DataFrame if possible
                    try:
                        return pd.DataFrame(result)
                    except:
                        return str(result)
                else:
                    return str(result)
            else:
                return "Execution error: No 'result' variable found."
                
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Execution error: {e}\nCode: {code}"
    
    def synthesize(self, question: str, df: pd.DataFrame, 
                  df_first_rows_hint: str = "") -> Optional[Dict[str, Any]]:
        """
        Main synthesis method that integrates with existing system.
        Returns a dictionary compatible with the existing query executor.
        """
        try:
            # Validate DataFrame for LangChain processing
            validation = validate_dataframe_for_langchain(df, self.profile)
            if not validation['is_valid']:
                logger.warning(f"DataFrame validation failed: {validation['errors']}")
            
            # Generate pandas code and execute
            result = self.llm_to_pandas(question, df)
            
            if isinstance(result, str) and "error" in result.lower():
                # Return error in expected format
                return {
                    "error": result,
                    "query_type": "error"
                }
            
            # Convert result to expected format
            return self._format_result_for_executor(result, question)
            
        except Exception as e:
            logger.error(f"LangChain synthesis failed: {e}")
            return {
                "error": f"LangChain synthesis failed: {e}",
                "query_type": "error"
            }
    
    def _format_result_for_executor(self, result: Union[pd.DataFrame, pd.Series], question: str) -> Dict[str, Any]:
        """Format the result for compatibility with existing query executor."""
        try:
            if isinstance(result, pd.DataFrame):
                return {
                    "filters": [],  # LangChain generates code directly, no explicit filters
                    "aggregations": self._extract_aggregations_from_result(result),
                    "sort_by": self._extract_sort_info_from_result(result),
                    "limit": len(result),
                    "query_type": "langchain_direct",
                    "result": result,
                    "langchain_generated": True
                }
            elif isinstance(result, pd.Series):
                # Convert Series to DataFrame
                df_result = result.to_frame()
                return {
                    "filters": [],
                    "aggregations": [],
                    "sort_by": [],
                    "limit": len(df_result),
                    "query_type": "langchain_series",
                    "result": df_result,
                    "langchain_generated": True
                }
            else:
                # Scalar result
                return {
                    "filters": [],
                    "aggregations": [],
                    "sort_by": [],
                    "limit": 1,
                    "query_type": "langchain_scalar",
                    "result": pd.DataFrame([{"value": result}]),
                    "langchain_generated": True
                }
                
        except Exception as e:
            logger.error(f"Failed to format result: {e}")
            return {
                "error": f"Failed to format result: {e}",
                "query_type": "error"
            }
    
    def _extract_aggregations_from_result(self, result: pd.DataFrame) -> List[Dict[str, str]]:
        """Extract aggregation information from the result (heuristic)."""
        # This is a simplified heuristic - in practice, this would need more sophisticated analysis
        aggregations = []
        
        # Look for common aggregation patterns in column names or data
        for col in result.columns:
            if any(agg in col.lower() for agg in ['sum', 'avg', 'mean', 'count', 'max', 'min']):
                aggregations.append({
                    "column": col,
                    "function": "unknown"  # Can't easily determine from result
                })
        
        return aggregations
    
    def _extract_sort_info_from_result(self, result: pd.DataFrame) -> List[Dict[str, str]]:
        """Extract sorting information from the result (heuristic)."""
        # This is a simplified heuristic - in practice, this would need more sophisticated analysis
        sort_info = []
        
        # Check if the result appears to be sorted (very basic check)
        if len(result) > 1:
            # This is a placeholder - real implementation would need to analyze the data
            pass
        
        return sort_info
