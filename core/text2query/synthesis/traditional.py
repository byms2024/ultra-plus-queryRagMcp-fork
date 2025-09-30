import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from config.logging_config import get_rag_logger
from config.settings import Config
from config.profiles import DataProfile
from ..utils.time_utils import parse_relative_date_range
from config.providers.registry import LLMFactory

logger = get_rag_logger()


class QuerySynthesizer:
    def __init__(self, config: Config, profile: DataProfile):
        self.config = config
        self.profile = profile
        
        # Create LLM provider using the profile's provider configuration
        provider_config = profile.get_provider_config()
        self.llm_provider = LLMFactory.create(provider_config)
        
        self.allowed_columns = profile.required_columns

    def extract_json_from_text(self, text: str) -> Optional[str]:
        try:
            text = text.strip()
            if text.startswith("```") and text.endswith("```"):
                first_newline = text.find('\n')
                last_backticks = text.rfind("```")
                if first_newline != -1 and last_backticks != -1:
                    text = text[first_newline + 1:last_backticks].strip()
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end + 1]
            return None
        except Exception:
            return None

    def synthesize(self, question: str, df_first_rows_hint: str = "") -> Optional[Dict[str, Any]]:
        try:
            window = parse_relative_date_range(question)
            window_hint = "none"
            if window is not None:
                ws, we = window
                window_hint = f"{ws.date()} to {we.date()}"

            system_rules = self.profile.get_llm_system_prompt()
            schema_hint = self.profile.get_schema_hints(df_first_rows_hint)

            prompt = (
                f"Question: {question}\n" +
                f"Detected date window hint: {window_hint}\n\n" +
                "Return only JSON matching the schema."
            )

            response = self.llm_provider.invoke(system_rules + "\n" + schema_hint + "\n" + prompt)
            # Handle LangChain response format
            if hasattr(response, 'content'):
                raw_text = response.content
            else:
                raw_text = str(response)
            json_str_extracted = self.extract_json_from_text(raw_text)
            json_str = json_str_extracted if isinstance(json_str_extracted, str) and json_str_extracted.strip() else raw_text
            if not json_str.strip():
                raise ValueError("No JSON content returned by model")
            spec = json.loads(json_str)

            if not isinstance(spec, dict):
                raise ValueError("Spec is not a JSON object")

            limit = int(spec.get('limit', 100))
            spec['limit'] = max(1, min(limit, 500))

            def _filter_cols(cols: Any) -> List[str]:
                return [c for c in (cols or []) if c in self.allowed_columns]

            if 'select' in spec:
                spec['select'] = _filter_cols(spec.get('select'))
            if 'group_by' in spec:
                spec['group_by'] = _filter_cols(spec.get('group_by'))

            if window is not None:
                has_date_filter = False
                date_columns = self.profile.date_columns
                for f in spec.get('filters', []) or []:
                    if isinstance(f, dict) and f.get('column') in date_columns:
                        has_date_filter = True
                        break
                if not has_date_filter and date_columns:
                    start, end = window
                    spec.setdefault('filters', []).append({
                        'column': date_columns[0], 'op': 'date_range', 'value': [str(start.date()), str(end.date())]
                    })

            print(spec)
            return spec

        except Exception as e:
            logger.warning(f"Failed to synthesize pandas spec: {e}")
            window = parse_relative_date_range(question)
            if window is not None and self.profile.date_columns:
                start, end = window
                return {
                    'filters': [
                        {'column': self.profile.date_columns[0], 'op': 'date_range', 'value': [str(start.date()), str(end.date())]}
                    ],
                    'limit': 100
                }
            return None


