"""Consolidated response building with formatting and statistics generation."""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, AsyncIterator
from config.profiles import DataProfile
from config.logging_config import get_rag_logger
from censor_utils.censoring import CensoringService
from config.providers.registry import LLMFactory, ProviderConfig
from config.base_config import load_system_config

logger = get_rag_logger()


class ResponseBuilder:
    """
    Consolidated response building with formatting and statistics generation.
    Combines response building, formatting, and statistics functionality.
    """
    
    def __init__(self, profile: DataProfile):
        self.profile = profile
        self.censor = CensoringService()
        self.visual_llm = self._initialize_visual_llm()
    
    def build_response(self, 
                      df_result: Optional[pd.DataFrame], 
                      query_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a complete response from query results."""
        
        if df_result is None or df_result.empty:
            return self._build_empty_response(query_spec)
        
        # Format for display and return
        table = self._format_dataframe_for_display(df_result)
        sources = self.profile.create_sources_from_df(df_result)
        
        narrative = self.generate_visual_summary(df_result, query_spec)

        response: Dict[str, Any] = {
            'answer': narrative if narrative else table,
            'sources': sources,
            'confidence': 'high',
            'query_spec': query_spec
        }

        if narrative:
            response['visual_answer'] = {'markdown': narrative}
            if table:
                response.setdefault('preview', {})['table_csv'] = table

        return response
    
    def _build_empty_response(self, query_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build response when no results are found."""
        empty_response = {
            'answer': 'No matching rows for your request.',
            'sources': [],
            'confidence': 'low',
            'query_spec': query_spec
        }

        if self.visual_llm:
            visual_summary = self._safe_visual_call(
                base_payload={
                    'summary': 'No matching data was returned for the requested filters.',
                    'row_count': 0,
                    'column_names': [],
                    'query_spec': query_spec or {}
                }
            )
            if visual_summary:
                empty_response['answer'] = visual_summary
                empty_response['visual_answer'] = {'markdown': visual_summary}

        return empty_response

    def _initialize_visual_llm(self):
        try:
            provider_config = self.profile.get_provider_config()
            # Allow profiles to opt-out by omitting generation model
            if not getattr(provider_config, 'generation_model', None):
                logger.info("Visual LLM skipped: no generation model configured")
                return None

            # Clone provider configuration to avoid mutating shared instance
            cloned_config = ProviderConfig(
                provider=provider_config.provider,
                generation_model=provider_config.generation_model,
                embedding_model=provider_config.embedding_model,
                credentials=dict(getattr(provider_config, 'credentials', {}) or {}),
                extras=dict(getattr(provider_config, 'extras', {}) or {}),
            )

            config = load_system_config()
            timeout = getattr(config, 'llm_request_timeout_seconds', 60)
            extras = cloned_config.extras
            if isinstance(extras, dict):
                extras.setdefault('temperature', 0.15)
                extras.setdefault('max_tokens', min(4096, extras.get('max_tokens', 4096)))

            logger.info("[visual] Initializing visual enrichment LLM")
            llm = LLMFactory.create(cloned_config)
            setattr(llm, '_visual_timeout', timeout)
            return llm
        except Exception as exc:
            logger.warning(f"Visual enrichment LLM unavailable: {exc}")
            return None

    def _describe_dataframe(self, df: pd.DataFrame, max_rows: int = 20) -> Tuple[str, int]:
        truncated = df.head(max_rows)
        description = truncated.to_json(orient='split')
        return description, len(df)

    def _construct_visual_prompt(self, df: pd.DataFrame, query_spec: Optional[Dict[str, Any]]) -> str:
        table_json, total_rows = self._describe_dataframe(df)
        columns_meta = []
        for column in df.columns:
            col_data = df[column]
            columns_meta.append({
                'name': column,
                'dtype': str(col_data.dtype),
                'sample_values': col_data.dropna().astype(str).head(5).tolist()
            })

        stats_snapshot = self.get_basic_stats(df)

        prompt_payload = {
            'instruction': (
                "Produce a plain Markdown report (<= 200 words) that presents the final result of the query. "
                "Use native Markdown elements like lists, tables and emoji icons when helpful, and use only H4 titles (####). "
                "Do not include JSON, code fences, or mermaid diagrams; respond with Markdown only."
            ),
            'query_spec': query_spec or {},
            'schema': columns_meta,
            'row_count': total_rows,
            'table_preview_json': table_json,
            'basic_stats': stats_snapshot
        }

        sanitized_payload = self._sanitize_for_json(prompt_payload)
        return json.dumps(sanitized_payload)

    def _safe_visual_call(self, base_payload: Dict[str, Any]) -> Optional[str]:
        if not self.visual_llm:
            return None

        start_time = time.perf_counter()
        try:
            timeout = getattr(self.visual_llm, '_visual_timeout', 60)
            instruction = (
                "Return a concise plain-Markdown summary of the provided context. "
                "Include the final result of the query and optionally Markdown tables, numbered or bulleted lists, and emoji icons to highlight insights."
                "Be precise and concise and use H4 titles (####) if you add headings."
                "Prefer numerated lists over tables unless the data is very large. Do not return JSON, code fences, or mermaid diagrams; respond with Markdown only."
            )

            sanitized_context = self._sanitize_for_json(base_payload)
            payload_dict = {'instruction': instruction, 'context': sanitized_context}
            sanitized_payload = self._sanitize_for_json(payload_dict)
            payload = json.dumps(sanitized_payload)

            raw_response = self.visual_llm.invoke(payload, config={'timeout': timeout})

            if hasattr(raw_response, 'content'):
                response_text = str(raw_response.content)
            else:
                response_text = str(raw_response)

            response_text = response_text.strip()

            if not response_text:
                duration = time.perf_counter() - start_time
                logger.warning("[visual] ⚠️ Visual LLM returned empty response after %.3f seconds", duration)
                return None

            duration = time.perf_counter() - start_time
            logger.info("[visual] ✅ Visual summary generated in %.3f seconds", duration)
            return response_text
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning(f"[visual] ❌ Visual enrichment failed after {duration:.3f} seconds: {exc}")

        return None

    def generate_visual_summary(
        self,
        df: Optional[pd.DataFrame],
        query_spec: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a user-friendly markdown summary with visuals based on DataFrame."""

        if df is None or df.empty:
            return None

        prompt = self._construct_visual_prompt(df, query_spec)

        base_payload = {
            'prompt': prompt,
            'query_spec': query_spec or {}
        }

        enriched = self._safe_visual_call(base_payload)
        if not enriched:
            return None

        return enriched
    
    async def generate_visual_summary_stream(
        self,
        df: Optional[pd.DataFrame],
        query_spec: Optional[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """
        Stream a user-friendly markdown summary with visuals based on DataFrame.
        Yields chunks of text as they are generated by the LLM.
        """
        if df is None or df.empty:
            yield ""
            return

        if not self.visual_llm:
            yield ""
            return

        start_time = time.perf_counter()
        
        try:
            prompt = self._construct_visual_prompt(df, query_spec)
            
            instruction = (
                "Return a concise plain-Markdown summary of the provided context. "
                "Include the final result of the query and optionally Markdown tables, numbered or bulleted lists, and emoji icons to highlight insights."
                "Be precise and concise and use H4 titles (####) if you add headings."
                "Prefer numerated lists over tables unless the data is very large. Do not return JSON, code fences, or mermaid diagrams; respond with Markdown only."
            )

            base_payload = {
                'prompt': prompt,
                'query_spec': query_spec or {}
            }
            
            sanitized_context = self._sanitize_for_json(base_payload)
            payload_dict = {'instruction': instruction, 'context': sanitized_context}
            sanitized_payload = self._sanitize_for_json(payload_dict)
            payload = json.dumps(sanitized_payload)

            timeout = getattr(self.visual_llm, '_visual_timeout', 60)
            
            # Stream from the LLM
            chunk_count = 0
            async for chunk in self.visual_llm.astream(payload, config={'timeout': timeout}):
                if hasattr(chunk, 'content'):
                    content = str(chunk.content)
                else:
                    content = str(chunk)
                
                if content:
                    chunk_count += 1
                    yield content
            
            duration = time.perf_counter() - start_time
            logger.info(f"[visual] ✅ Visual summary streamed in {duration:.3f} seconds ({chunk_count} chunks)")
            
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning(f"[visual] ❌ Visual enrichment streaming failed after {duration:.3f} seconds: {exc}")
            yield ""

    def _sanitize_for_json(self, value: Any) -> Any:
        """Recursively convert values into JSON-serializable primitives."""

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if isinstance(value, (np.generic,)):
            return value.item()

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, pd.Series):
            return [self._sanitize_for_json(v) for v in value.tolist()]

        if isinstance(value, pd.Index):
            return [self._sanitize_for_json(v) for v in value.tolist()]

        if isinstance(value, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in value]

        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                return value.item()
            except Exception:
                pass

        return str(value)
    
    def _format_dataframe_for_display(self, 
                                    df: pd.DataFrame, 
                                    max_rows: int = 50, 
                                    max_chars: int = 6000) -> str:
        """Format a DataFrame into a compact CSV-like table string for display."""
        try:
            if len(df) > max_rows:
                df = df.head(max_rows)
            
            table = df.to_csv(index=False)
            
            # If still too large, reduce rows further
            while len(table) > max_chars and len(df) > 5:
                df = df.head(max(5, len(df) // 2))
                table = df.to_csv(index=False)
            
            logger.debug("Formatted DataFrame for display")
            return table
            
        except Exception as e:
            logger.warning(f"Failed to format DataFrame for display: {e}")
            return ""
    
    def format_dataframe_for_prompt(self, 
                                  df: pd.DataFrame, 
                                  max_rows: int = 50, 
                                  max_chars: int = 6000) -> str:
        """Format a DataFrame into a compact CSV-like table string for prompting."""
        try:
            if len(df) > max_rows:
                df = df.head(max_rows)
            table = df.to_csv(index=False)
            # If still too large, reduce rows further
            while len(table) > max_chars and len(df) > 5:
                df = df.head(max(5, len(df) // 2))
                table = df.to_csv(index=False)
            return table
        except Exception as e:
            logger.warning(f"Failed to format DataFrame for prompt: {e}")
            return ""
    
    # Statistics generation methods (from StatsGenerator)
    def generate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics for the dataset."""
        stats_columns = self.profile.get_stats_columns()
        
        stats = {'total_records': len(df)}
        
        # Generate stats based on profile configuration
        for stat_name, column in stats_columns.items():
            if column in df.columns:
                if stat_name == 'dealers_count':
                    stats[stat_name] = df[column].nunique()
                elif stat_name == 'average_score':
                    stats[stat_name] = float(df[column].mean())
                elif stat_name == 'repair_types':
                    stats[stat_name] = df[column].value_counts().head(10).to_dict()
                elif stat_name == 'date_range':
                    stats[stat_name] = {
                        'earliest': str(df[column].min()),
                        'latest': str(df[column].max())
                    }
        
        # Add censoring statistics
        stats['censor_stats'] = self.get_censor_stats()
        return stats
    
    def get_censor_stats(self) -> Dict[str, Any]:
        """Get statistics about data censoring operations."""
        return self.censor.get_stats()
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict()
        }
    
    def get_column_stats(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific column."""
        if column not in df.columns:
            return {'error': f'Column {column} not found'}
        
        col_data = df[column]
        stats = {
            'column_name': column,
            'dtype': str(col_data.dtype),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique()
        }
        
        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'mean': float(col_data.mean()) if not col_data.empty else None,
                'median': float(col_data.median()) if not col_data.empty else None,
                'std': float(col_data.std()) if not col_data.empty else None,
                'min': float(col_data.min()) if not col_data.empty else None,
                'max': float(col_data.max()) if not col_data.empty else None
            })
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            stats.update({
                'earliest': str(col_data.min()) if not col_data.empty else None,
                'latest': str(col_data.max()) if not col_data.empty else None
            })
        else:
            # For categorical/text columns
            value_counts = col_data.value_counts().head(10)
            stats['top_values'] = value_counts.to_dict()
        
        return stats


# Standalone functions for backward compatibility
def format_dataframe_for_prompt(df: pd.DataFrame, max_rows: int = 50, max_chars: int = 6000) -> str:
    """Standalone function for formatting DataFrame for prompts (backward compatibility)."""
    try:
        if len(df) > max_rows:
            df = df.head(max_rows)
        table = df.to_csv(index=False)
        while len(table) > max_chars and len(df) > 5:
            df = df.head(max(5, len(df) // 2))
            table = df.to_csv(index=False)
        return table
    except Exception as e:
        logger.warning(f"Failed to format DataFrame for prompt: {e}")
        return ""


def create_sources_from_df(df: pd.DataFrame, limit: int = 20) -> List[Dict[str, Any]]:
    """
    DEPRECATED: This function has been moved to profile-specific implementations.
    Use profile.create_sources_from_df() instead for profile-specific source creation.
    """
    logger.warning("create_sources_from_df() standalone function is deprecated. Use profile.create_sources_from_df() instead.")
    
    # Fallback generic implementation
    sources: List[Dict[str, Any]] = []
    cols = set(df.columns)
    take = min(limit, len(df))
    
    for i in range(take):
        row = df.iloc[i]
        source = {}
        for col in cols:
            # Generic source creation - just include all columns
            if pd.notna(row[col]):
                source[col.lower()] = str(row[col])
            else:
                source[col.lower()] = ''
        sources.append(source)
    
    return sources


# Backward compatibility alias
StatsGenerator = ResponseBuilder