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
        
        # Format for display and return (no table/preview formatting needed)
        sources = self.profile.create_sources_from_df(df_result)
        
        narrative = self.generate_visual_summary(df_result, query_spec)

        response: Dict[str, Any] = {
            'answer': narrative,
            'sources': sources,
            'confidence': 'high',
            'query_spec': query_spec
        }

        return response
    
    def _detect_language_from_question(self, question: Optional[str]) -> str:
        """Very lightweight language detection for EN/PT; defaults to EN."""
        if not question:
            return "en"
        q = (question or "").lower()
        # Chinese characters detection (CJK Unified Ideographs and Extension A)
        try:
            if any('\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' for ch in question):
                return "zh"
        except Exception:
            pass
        # Portuguese indicators
        pt_tokens = [" que ", " como ", " por que", " qual ", " quais ", " são ", " nao ", "não ", " quantos", " média", " soma "]
        if any(tok in q for tok in pt_tokens) or any(ch in q for ch in "ãõáéíóúçâêô" ):
            return "pt"
        return "en"

    def _localize(self, text_id: str, lang: str) -> str:
        """Return a localized string for small set of defaults."""
        catalog = {
            'no_rows_title_en': 'No matching rows for your request.',
            'no_rows_title_pt': 'Nenhuma linha correspondente para sua solicitação.',
            'no_rows_title_zh': '未找到与您的请求匹配的行。',
            'no_rows_summary_en': 'No matching data was returned for the requested filters.',
            'no_rows_summary_pt': 'Nenhum dado correspondente foi retornado para os filtros solicitados.',
            'no_rows_summary_zh': '根据所选筛选条件，没有返回匹配的数据。',
        }
        key = f"{text_id}_{'pt' if lang=='pt' else 'en'}"
        if lang == 'zh':
            key = f"{text_id}_zh"
        return catalog.get(key, catalog.get(f"{text_id}_en", ""))

    def _build_empty_response(self, query_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build response when no results are found."""
        question = None
        if isinstance(query_spec, dict):
            question = query_spec.get('question') or getattr(self, 'last_question', None)
        lang = self._detect_language_from_question(question)

        empty_response = {
            'answer': self._localize('no_rows_title', lang),
            'sources': [],
            'confidence': 'low',
            'query_spec': query_spec
        }

        if self.visual_llm:
            base_summary = self._localize('no_rows_summary', lang)
            visual_summary = self._safe_visual_call(
                base_payload={
                    'summary': base_summary,
                    'row_count': 0,
                    'column_names': [],
                    'query_spec': query_spec or {}
                }
            )
            if visual_summary:
                empty_response['answer'] = visual_summary

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
            instruction = self._get_visual_markdown_instruction()

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

    def _get_visual_markdown_instruction(self, language_hint: str, margin_lg: int = 16, margin_sm: int = 8) -> str:
        """Single source of truth for visual markdown instruction to avoid duplication."""
        return (
            "Return a concise plain-Markdown including the final result of the query, without mentioning the query or the data."
            "Use Markdown tables, numbered or bulleted lists"
            "Always use emojis to highlight."
            f"Be precise and concise and use H5 titles (#####), after the title use a margin of {margin_lg}px."
            f"Use a margin of {margin_sm}px anywhere else in the content."
            "Prefer numerated lists over tables unless the data is very large. Do not return JSON or code fences."
            "Do not include any other text or explanation."
            f"Answer in this language: {language_hint}."
            
        )

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
            return 

        return enriched
    
    async def generate_visual_summary_stream(
        self,
        df: Optional[pd.DataFrame],
        query_spec: Optional[Dict[str, Any]],
        question: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream a user-friendly markdown summary with visuals based on DataFrame.
        Yields chunks of text as they are generated by the LLM.
        """
        lang = self._detect_language_from_question(question)

        if df is None or df.empty:            
            yield self._localize('no_rows_title', lang)
            return

        if not self.visual_llm:
            yield "Error: Visual LLM not available."
            return

        start_time = time.perf_counter()

        try:
            prompt = self._construct_visual_prompt(df, query_spec)

            # Use localized instruction that includes language awareness
            instruction = self._get_visual_markdown_instruction(lang)

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
            yield "Error: Visual enrichment streaming failed."

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