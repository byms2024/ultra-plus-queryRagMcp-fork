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
        """Delegate language detection to profile."""
        detect = getattr(self.profile, 'detect_language', None)
        result = detect(question) if callable(detect) else "en"
        logger.info(f"[language] 🔍 Profile language detection: '{question}' → '{result}'")
        return result

    def _localize(self, text_id: str, lang: str) -> str:
        """Delegate localization to profile."""
        localize = getattr(self.profile, 'localize', None)
        return localize(text_id, lang) if callable(localize) else ""

    def _build_empty_response(self, query_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build response when no results are found."""
        question = None
        if isinstance(query_spec, dict):
            question = query_spec.get('question') or getattr(self, 'last_question', None)
        lang = self._detect_language_from_question(question)

        # Start with basic fallback response
        fallback_answer = self._localize('no_rows_title', lang)
        
        empty_response = {
            'answer': fallback_answer,
            'sources': [],
            'confidence': 'low',
            'query_spec': query_spec
        }

        # If visual LLM is available, generate contextual empty response
        if self.visual_llm and question:
            visual_summary = self._generate_contextual_empty_response(
                question=question,
                query_spec=query_spec,
                language=lang
            )
            if visual_summary:
                empty_response['answer'] = visual_summary
        elif self.visual_llm:
            # No question available, use basic visual enrichment
            base_summary = self._localize('no_rows_summary', lang)
            visual_summary = self._safe_visual_call(
                base_payload={
                    'summary': base_summary,
                    'row_count': 0,
                    'column_names': [],
                    'query_spec': query_spec or {}
                },
                language_hint=lang
            )
            if visual_summary:
                empty_response['answer'] = visual_summary

        return empty_response

    def _generate_contextual_empty_response(
        self, 
        question: str, 
        query_spec: Optional[Dict[str, Any]], 
        language: str
    ) -> Optional[str]:
        """
        Generate an intelligent, contextual empty response based on the user's question.
        
        This method uses the visual LLM to create a user-friendly explanation of why
        no results were found, considering the context of the query.
        """
        if not self.visual_llm:
            return None
        
        start_time = time.perf_counter()
        
        try:
            # Extract query context for better response generation
            filters = query_spec.get('filters', {}) if query_spec else {}
            aggregations = query_spec.get('aggregations', []) if query_spec else []
            sort_by = query_spec.get('sort_by') if query_spec else None
            
            # Build context payload with question and query details
            context_payload = {
                'question': question,
                'row_count': 0,
                'query_spec': query_spec or {},
                'filters_applied': filters,
                'aggregations_requested': aggregations,
                'sort_requested': sort_by,
                'result_status': 'empty'
            }
            
            # Create intelligent instruction for empty response generation
            empty_response_instruction = self._get_empty_response_instruction(language)
            
            sanitized_context = self._sanitize_for_json(context_payload)
            payload_dict = {
                'instruction': empty_response_instruction,
                'context': sanitized_context
            }
            sanitized_payload = self._sanitize_for_json(payload_dict)
            payload = json.dumps(sanitized_payload, separators=(',', ':'))  # Compact JSON
            
            timeout = getattr(self.visual_llm, '_visual_timeout', 60)
            
            # OPTIMIZATION: Use shorter timeout and token limit for empty responses
            # Empty responses should be quick and concise
            config_params = {
                'timeout': min(timeout, 15),  # Max 15 seconds for empty responses
                'max_tokens': 300,  # Limit to 300 tokens for concise responses
                'stop': ['\n\n\n', '---END---']
            }
            
            raw_response = self.visual_llm.invoke(payload, config=config_params)
            
            if hasattr(raw_response, 'content'):
                response_text = str(raw_response.content)
            else:
                response_text = str(raw_response)
            
            response_text = response_text.strip()
            
            if response_text:
                duration = time.perf_counter() - start_time
                logger.info("[empty_response] ✅ Contextual empty response generated in %.3f seconds", duration)
                return response_text
            else:
                duration = time.perf_counter() - start_time
                logger.warning("[empty_response] ⚠️ Visual LLM returned empty response after %.3f seconds", duration)
                return None
                
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning(f"[empty_response] ❌ Failed to generate contextual empty response after {duration:.3f} seconds: {exc}")
            return None
    
    def _get_empty_response_instruction(self, language: str) -> str:
        """
        Get instruction for generating contextual empty responses.
        Profiles can override this for customization.
        """
        # Try to get profile-specific instruction if available
        profile_instruction_fn = getattr(self.profile, 'get_empty_response_instruction', None)
        if callable(profile_instruction_fn):
            return profile_instruction_fn(language)
        
        # Default instruction for empty responses
        language_map = {
            'pt': 'Portuguese (Brazilian)',
            'zh': 'Chinese (Simplified)',
            'en': 'English'
        }
        lang_name = language_map.get(language, 'English')
        
        return f"""You are an intelligent data assistant helping users understand why their query returned no results.

Based on the user's question and query context provided, generate a friendly, helpful response that:

1. **Acknowledges** the empty result in a natural way
2. **Explains** possible reasons why no data was found (e.g., filters too restrictive, date range outside data coverage, specific values not existing)
3. **Suggests** helpful next steps or alternative queries the user might try
4. **References** specific filters or conditions from the query when relevant (e.g., dealer codes, date ranges, score thresholds)
5. **Maintains context** by incorporating terms and concepts from the original question

**Formatting Requirements:**
- Use clear, conversational language in {lang_name}
- Use emojis sparingly but effectively (1-2 emojis maximum)
- Keep response concise (2-4 sentences)
- Use H5 headings (####) if structuring with sections
- Do NOT use technical jargon like "null", "empty DataFrame", "no rows"
- Do NOT return code or JSON
- Be empathetic and helpful, not robotic

**Example good responses:**

For "What's the NPS for dealer ABC in January?":
"##### 📊 No Data Found

Unfortunately, there are no records for dealer ABC during January. This could mean the dealer had no service orders in that period, or the data hasn't been uploaded yet. Try checking a broader date range or verifying the dealer code."

For "Show me promoters with score above 9":
"##### 😊 No Matching Customers

We didn't find any customers matching your criteria. Remember that promoters have scores of 9-10, so filtering for scores above 9 will only show customers with a perfect 10. You might want to search for scores >= 9 instead."

Now generate an appropriate response based on the context provided."""
    
    async def _generate_contextual_empty_response_stream(
        self,
        question: str,
        query_spec: Optional[Dict[str, Any]],
        language: str
    ) -> AsyncIterator[str]:
        """
        Stream an intelligent, contextual empty response based on the user's question.
        
        This async method streams chunks of the empty response as they are generated.
        """
        if not self.visual_llm:
            return
        
        start_time = time.perf_counter()
        
        try:
            # Extract query context for better response generation
            filters = query_spec.get('filters', {}) if query_spec else {}
            aggregations = query_spec.get('aggregations', []) if query_spec else []
            sort_by = query_spec.get('sort_by') if query_spec else None
            
            # Build context payload with question and query details
            context_payload = {
                'question': question,
                'row_count': 0,
                'query_spec': query_spec or {},
                'filters_applied': filters,
                'aggregations_requested': aggregations,
                'sort_requested': sort_by,
                'result_status': 'empty'
            }
            
            # Create intelligent instruction for empty response generation
            empty_response_instruction = self._get_empty_response_instruction(language)
            
            sanitized_context = self._sanitize_for_json(context_payload)
            payload_dict = {
                'instruction': empty_response_instruction,
                'context': sanitized_context
            }
            sanitized_payload = self._sanitize_for_json(payload_dict)
            payload = json.dumps(sanitized_payload, separators=(',', ':'))  # Compact JSON
            
            timeout = getattr(self.visual_llm, '_visual_timeout', 60)
            
            # OPTIMIZATION: Faster config for streaming empty responses
            config_params = {
                'timeout': min(timeout, 15),  # Quick timeout for empty responses
                'max_tokens': 300,  # Concise responses
                'stop': ['\n\n\n', '---END---']
            }
            
            # Stream from the LLM
            chunk_count = 0
            async for chunk in self.visual_llm.astream(payload, config=config_params):
                if hasattr(chunk, 'content'):
                    content = str(chunk.content)
                else:
                    content = str(chunk)
                
                if content:
                    chunk_count += 1
                    yield content
            
            duration = time.perf_counter() - start_time
            logger.info(f"[empty_response] ✅ Contextual empty response streamed in {duration:.3f} seconds ({chunk_count} chunks)")
            
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning(f"[empty_response] ❌ Failed to stream contextual empty response after {duration:.3f} seconds: {exc}")
            # Yield fallback message on error
            yield self._localize('no_rows_title', language)

    def _initialize_visual_llm(self):
        try:
            provider_config = self.profile.get_provider_config()
            # Allow profiles to opt-out by omitting generation model
            if not getattr(provider_config, 'generation_model', None):
                logger.info("Visual LLM skipped: no generation model configured")
                return None

            # Clone provider configuration with optimizations
            extras = dict(getattr(provider_config, 'extras', {}) or {})
            
            # OPTIMIZATION: Increase temperature slightly for faster token sampling
            # Lower temperatures require more computation for probability distribution
            if 'temperature' not in extras or extras['temperature'] < 0.3:
                extras['temperature'] = 0.3
            
            # OPTIMIZATION: Add response format hints to speed up parsing
            # Some models can optimize when they know the expected format
            extras['response_format_hint'] = 'markdown'
            
            cloned_config = ProviderConfig(
                provider=provider_config.provider,
                generation_model=provider_config.generation_model,
                embedding_model=provider_config.embedding_model,
                credentials=dict(getattr(provider_config, 'credentials', {}) or {}),
                extras=extras,
            )

            config = load_system_config()
            # OPTIMIZATION: Reduce timeout for faster failures (30s instead of 60s)
            # Visual summaries should be fast; slow responses indicate issues
            timeout = min(getattr(config, 'llm_request_timeout_seconds', 60), 30)
            # Do not set internal defaults; rely on profile/provider config

            logger.info("[visual] Initializing visual enrichment LLM with optimizations")
            llm = LLMFactory.create(cloned_config)
            setattr(llm, '_visual_timeout', timeout)
            setattr(llm, '_optimized', True)  # Flag for monitoring
            return llm
        except Exception as exc:
            logger.warning(f"Visual enrichment LLM unavailable: {exc}")
            return None

    def _describe_dataframe(self, df: pd.DataFrame, max_rows: int = 20) -> Tuple[str, int]:
        """
        OPTIMIZATION: Use CSV instead of JSON for more compact representation.
        CSV is typically 30-40% smaller than JSON for tabular data and faster to parse.
        """
        truncated = df.head(max_rows)
        # Use CSV without index for compact representation
        description = truncated.to_csv(index=False, lineterminator='\n')
        return description, len(df)

    def _construct_visual_prompt(self, df: pd.DataFrame, query_spec: Optional[Dict[str, Any]], max_rows: int = 20) -> str:
        """
        OPTIMIZATION: Reduce prompt payload size by minimizing redundant information
        and using more compact data representation.
        """
        table_preview, total_rows = self._describe_dataframe(df, max_rows=max_rows)
        
        # OPTIMIZATION: Only include essential column metadata, skip sample values
        # Sample values are already visible in table preview
        columns_meta = [
            {'name': col, 'dtype': str(df[col].dtype)}
            for col in df.columns
        ]

        # OPTIMIZATION: Only include essential stats (skip verbose details)
        essential_stats = {
            'row_count': total_rows,
            'column_count': len(df.columns),
            'columns': list(df.columns)
        }

        # OPTIMIZATION: Simplify query_spec to only include relevant fields
        simplified_query_spec = {}
        if query_spec:
            for key in ['question', 'filters', 'aggregations']:
                if key in query_spec and query_spec[key]:
                    simplified_query_spec[key] = query_spec[key]

        prompt_payload = {
            'query_spec': simplified_query_spec,
            'schema': columns_meta,
            'stats': essential_stats,
            'data_preview': table_preview  # CSV format, much more compact
        }

        sanitized_payload = self._sanitize_for_json(prompt_payload)
        return json.dumps(sanitized_payload, separators=(',', ':'))  # Compact JSON (no spaces)

    def _safe_visual_call(self, base_payload: Dict[str, Any], language_hint: Optional[str] = None) -> Optional[str]:
        if not self.visual_llm:
            return None

        start_time = time.perf_counter()
        try:
            timeout = getattr(self.visual_llm, '_visual_timeout', 60)
            # Require profile-defined visual instruction; do not fallback to internal
            instruction_provider = getattr(self.profile, 'get_visual_markdown_instruction', None)
            if not callable(instruction_provider):
                logger.error("[visual] Missing profile.get_visual_markdown_instruction; skipping visual enrichment")
                return None
            # Ensure we have a valid language hint
            final_lang = language_hint or "en"
            logger.info(f"[visual] 🎯 Using language: '{final_lang}' (hint was: '{language_hint}')")
            instruction = instruction_provider(final_lang)

            sanitized_context = self._sanitize_for_json(base_payload)
            
            # OPTIMIZATION: Compact payload with minimal formatting
            payload_dict = {'instruction': instruction, 'context': sanitized_context}
            sanitized_payload = self._sanitize_for_json(payload_dict)
            payload = json.dumps(sanitized_payload, separators=(',', ':'))  # No whitespace

            # OPTIMIZATION: Add stop sequences to prevent over-generation
            config_params = {
                'timeout': timeout,
                'stop': ['\n\n\n', '---END---']  # Stop on excessive whitespace
            }
            
            raw_response = self.visual_llm.invoke(payload, config=config_params)

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

    # Removed internal visual instruction template; profiles must provide get_visual_markdown_instruction

    def generate_visual_summary(
        self,
        df: Optional[pd.DataFrame],
        query_spec: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a user-friendly markdown summary with visuals based on DataFrame."""

        if df is None or df.empty:
            return None

        prompt = self._construct_visual_prompt(df, query_spec)

        # Extract question for prominent placement in context
        question = None
        if isinstance(query_spec, dict):
            question = query_spec.get('question')
        
        base_payload = {
            'user_question': question,  # Make question prominent at top level
            'prompt': prompt,
            'query_spec': query_spec or {}
        }

        # Determine language from question if available
        lang = "en"  # Default fallback
        if isinstance(query_spec, dict):
            question = query_spec.get('question')
            logger.info(f"[visual] 🔍 Language detection - query_spec type: {type(query_spec)}, question: '{question}'")
            if question:  # Only detect if question exists
                lang = self._detect_language_from_question(question)
                logger.info(f"[visual] 🌐 Detected language: '{lang}' for question: '{question}'")

        enriched = self._safe_visual_call(base_payload, language_hint=lang)
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
            # Stream intelligent empty response if question is available
            if self.visual_llm and question:
                async for chunk in self._generate_contextual_empty_response_stream(
                    question=question,
                    query_spec=query_spec,
                    language=lang
                ):
                    yield chunk
            else:
                # Fallback to basic localized message
                yield self._localize('no_rows_title', lang)
            return

        if not self.visual_llm:
            yield "Error: Visual LLM not available."
            return

        start_time = time.perf_counter()

        try:
            prompt = self._construct_visual_prompt(df, query_spec, max_rows=8)

            # Require profile-defined instruction; no internal fallback
            instruction_provider = getattr(self.profile, 'get_visual_markdown_instruction', None)
            if not callable(instruction_provider):
                logger.error("[visual] Missing profile.get_visual_markdown_instruction; streaming disabled")
                yield "Error: Visual LLM not available."
                return
            instruction = instruction_provider(lang)

            # Extract question for prominent placement in context
            user_question = None
            if isinstance(query_spec, dict):
                user_question = query_spec.get('question')
            
            base_payload = {
                'user_question': user_question,  # Make question prominent at top level
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
    # Note: generic create_sources_from_df removed; all source creation must be profile-defined


# Backward compatibility alias removed; use ResponseBuilder for stats via its methods