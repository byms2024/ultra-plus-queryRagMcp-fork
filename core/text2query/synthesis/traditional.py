import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

from config.logging_config import get_rag_logger
from config.base_config import Config, load_system_config
from config.profiles import DataProfile
from ..utils.time_utils import parse_relative_date_range, StepTimer
from ..data import build_schema_description, validate_dataframe_for_langchain
from config.providers.registry import LLMFactory

logger = get_rag_logger()


class QuerySynthesizer:
    def __init__(self, config: Config, profile: DataProfile):
        self.config = config
        self.profile = profile
        self.strategy_name = "traditional"
        
        # Create LLM provider using the profile's provider configuration
        provider_config = profile.get_provider_config()
        self.llm_provider = LLMFactory.create(provider_config)
        
        self.allowed_columns = profile.required_columns

    # ---- Pandas-code generation helpers (kept local for simplicity) ----
    def _build_system_prompt(self) -> str:
        try:
            base_prompt = self.profile.get_llm_system_prompt()
            instructions = (
                "You are a Python data assistant specialized in generating pandas code.\n"
                "Generate ONLY pandas code and assign the final result to a variable named 'result'.\n"
                "Use the provided DataFrame 'df' as the data source."
            )
            return f"{base_prompt}\n\n{instructions}"
        except Exception:
            return (
                "You are a Python data assistant. Generate pandas code to answer the user's question. "
                "Assign the final output to 'result'."
            )

    def _build_schema_hints(self, schema_description: str) -> str:
        try:
            base_hints = self.profile.get_schema_hints(schema_description)
            extra = (
                "Use only the columns listed above. Respect data types. "
                "Compare date columns using proper datetime operations."
            )
            return f"{base_hints}\n\nSCHEMA INFORMATION:\n{schema_description}\n\n{extra}"
        except Exception:
            return f"Available columns: {', '.join(self.allowed_columns)}"

    def _handle_date_context(self, query: str) -> str:
        try:
            window = parse_relative_date_range(query)
            if window is not None:
                start_date, end_date = window
                return (
                    "DATE CONTEXT:\n"
                    f"- Start Date: {start_date.date()}\n"
                    f"- End Date: {end_date.date()}\n"
                )
            return ""
        except Exception:
            return ""

    def _build_complete_prompt(self, query: str, schema_description: str) -> str:
        parts = [
            self._build_system_prompt(),
            self._build_schema_hints(schema_description),
            self._handle_date_context(query),
            f"USER QUESTION: {query}",
            "Generate pandas code that assigns the result to a variable named 'result'.",
        ]
        return "\n\n".join(filter(None, parts))

    def _extract_code_from_response(self, response) -> str:
        # Accept Provider or raw string
        code = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        # Remove fenced blocks
        if code.startswith("```") and code.endswith("```"):
            lines = code.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "````":
                lines = lines[:-1]
            elif lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        # Remove language tags
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```py"):
            code = code[5:]
        code = code.strip()
        # Harden: strip import lines and dunder usage
        filtered_lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            if '__import__' in stripped or 'eval(' in stripped or 'exec(' in stripped:
                continue
            filtered_lines.append(line)
        code = '\n'.join(filtered_lines).strip()
        if "result =" not in code:
            raise ValueError("Generated code must assign the final output to a variable named 'result'.")
        return code

    def _execute_pandas_code(self, code: str, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Any]:
        # Restricted execution environment; use same dict for globals/locals so functions see symbols like 'pd'
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
                'round': round,
            },
        }
        exec(code, env, env)
        if 'result' not in env:
            raise RuntimeError("Execution error: No 'result' variable found.")
        return env['result']

    def _format_result_for_executor(self, result: Union[pd.DataFrame, pd.Series, Any]) -> Dict[str, Any]:
        if isinstance(result, pd.DataFrame):
            return {
                "filters": [],
                "aggregations": [],
                "sort_by": [],
                "limit": len(result),
                "query_type": "traditional_direct",
                "result": result,
                "langchain_generated": False,
            }
        if isinstance(result, pd.Series):
            return {
                "filters": [],
                "aggregations": [],
                "sort_by": [],
                "limit": len(result),
                "query_type": "traditional_series",
                "result": result.to_frame(),
                "langchain_generated": False,
            }
        # Scalar or other iterable -> try DataFrame, else stringify
        try:
            df = pd.DataFrame(result) if not pd.api.types.is_scalar(result) else pd.DataFrame([{"value": result}])
        except Exception:
            df = pd.DataFrame([{"value": str(result)}])
        return {
            "filters": [],
            "aggregations": [],
            "sort_by": [],
            "limit": len(df),
            "query_type": "traditional_scalar",
            "result": df,
            "langchain_generated": False,
        }

    # ---- Public API ----
    def synthesize(self, question: str, df: pd.DataFrame, df_first_rows_hint: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate pandas code for the question, execute it, and return a result dict
        compatible with the executor/response builder (direct result, like LangChain).
        """
        try:
            # Validate DF for processing (re-using existing helper)
            with StepTimer(logger, f"[{self.strategy_name}] Validate DataFrame", "🧪"):
                validation = validate_dataframe_for_langchain(df, self.profile)
            if not validation.get('is_valid', True):
                logger.warning(f"[{self.strategy_name}] DataFrame validation warnings: {validation.get('errors')}")

            # Build prompt
            with StepTimer(logger, f"[{self.strategy_name}] Build schema description", "🧬"):
                schema_description = build_schema_description(df, self.profile)
            with StepTimer(logger, f"[{self.strategy_name}] Build complete prompt", "🧩"):
                full_prompt = self._build_complete_prompt(question, schema_description)

            # Invoke LLM with timeout using a thread wrapper (to keep parity with prior behavior)
            llm_timeout = getattr(load_system_config(), "llm_request_timeout_seconds", 60)
            try:
                with StepTimer(logger, f"[{self.strategy_name}] LLM call (direct pandas)", "🤖"):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.llm_provider.invoke, full_prompt)
                        response = future.result(timeout=llm_timeout)
            except FuturesTimeoutError:
                logger.warning(f"[{self.strategy_name}] ⏱️ LLM call timed out after {llm_timeout:.2f}s")
                return {
                    "error": f"LLM call timed out after {llm_timeout:.2f}s",
                    "query_type": "error",
                }

            # Extract code and execute safely
            with StepTimer(logger, f"[{self.strategy_name}] Extract generated code", "📝"):
                code = self._extract_code_from_response(response)
            logger.debug(f"[{self.strategy_name}] Generated pandas code: {code}")
            with StepTimer(logger, f"[{self.strategy_name}] Execute pandas code", "⚙️"):
                result_obj = self._execute_pandas_code(code, df)

            with StepTimer(logger, f"[{self.strategy_name}] Format result for executor", "📦"):
                formatted = self._format_result_for_executor(result_obj)
            return formatted

        except Exception as e:
            logger.error(f"[{self.strategy_name}] ❌ Pandas generation/execution failed: {e}")
            return {
                "error": f"Traditional direct synthesis failed: {e}",
                "query_type": "error",
            }


