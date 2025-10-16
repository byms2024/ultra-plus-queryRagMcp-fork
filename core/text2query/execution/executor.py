from typing import Any, Dict, List, cast
import pandas as pd
from config.logging_config import get_rag_logger
from config.profiles import DataProfile

logger = get_rag_logger()


class QueryExecutor:
    def __init__(self, profile: DataProfile):
        self.profile = profile
    
    def apply(self, df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
        local = df.copy()

        def apply_filter(local_df: pd.DataFrame, flt: Dict[str, Any]) -> pd.DataFrame:
            column = flt.get('column')
            op = str(flt.get('op', '')).lower()
            value = flt.get('value')
            if column not in local_df.columns:
                return local_df
            series = local_df[column]
            if op == 'eq':
                return local_df.loc[series == value]
            if op == 'neq':
                return local_df.loc[series != value]
            if op == 'gt':
                return local_df.loc[series > value]
            if op == 'gte':
                return local_df.loc[series >= value]
            if op == 'lt':
                return local_df.loc[series < value]
            if op == 'lte':
                return local_df.loc[series <= value]
            if op == 'in':
                try:
                    vals = value if isinstance(value, list) else [value]
                    return local_df.loc[series.isin(vals)]
                except Exception:
                    return local_df
            if op == 'contains':
                try:
                    return local_df.loc[series.astype(str).str.contains(str(value), case=False, na=False)]
                except Exception:
                    return local_df
            if op == 'date_range':
                try:
                    start_s, end_s = value if isinstance(value, (list, tuple)) and len(value) == 2 else (None, None)
                    if start_s is None or end_s is None:
                        return local_df
                    start_dt = pd.to_datetime(start_s, errors='coerce')
                    end_dt = pd.to_datetime(end_s, errors='coerce')
                    if column in self.profile.date_columns:
                        series_dt = pd.to_datetime(series, errors='coerce')
                        return local_df.loc[(series_dt >= start_dt) & (series_dt <= end_dt)]
                    return local_df
                except Exception:
                    return local_df
            return local_df

        # Filters
        for flt in spec.get('filters', []) or []:
            if isinstance(flt, dict):
                local = apply_filter(local, flt)

        # Group by and aggregations
        group_by = spec.get('group_by') or []
        aggregations = spec.get('aggregations') or {}
        if group_by and aggregations:
            try:
                local = local.groupby(group_by, as_index=False).agg(aggregations)
                flat_cols: List[str] = []
                for col in local.columns:
                    if isinstance(col, tuple):
                        flat_cols.append("_".join([str(c) for c in col if c]))
                    else:
                        flat_cols.append(str(col))
                local.columns = flat_cols
            except Exception as e:
                logger.warning(f"Failed group/agg: {e}")

        # Select
        select_cols = spec.get('select') or []
        if select_cols:
            existing = [c for c in select_cols if c in local.columns]
            if existing:
                local = local[existing]

        # Sort
        sort_spec = spec.get('sort') or []
        if isinstance(sort_spec, list) and sort_spec and isinstance(local, pd.DataFrame):
            try:
                by_cols = [s.get('by') for s in sort_spec if isinstance(s, dict) and s.get('by') in local.columns]
                orders = [str(s.get('order', 'asc')).lower() != 'desc' for s in sort_spec if isinstance(s, dict) and s.get('by') in local.columns]
                if by_cols:
                    local = cast(pd.DataFrame, local.sort_values(by=by_cols, ascending=orders))
            except Exception as e:
                logger.warning(f"Failed sort: {e}")

        # Limit
        try:
            limit = int(spec.get('limit', 20))
        except Exception:
            limit = 20
        limit = max(1, min(limit, 30))
        if isinstance(local, pd.DataFrame):
            local = local.head(limit)
        return cast(pd.DataFrame, local)


