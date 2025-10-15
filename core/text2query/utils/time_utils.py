"""Consolidated time and date utilities with enhanced functionality."""

from typing import Optional, Tuple, cast, Union, List, Dict, Any
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

class StepTimer:
    """Context manager to log start/end and duration of a processing step.

    Usage:
        with StepTimer(logger, "Build prompt", emoji="🧩"):
            do_work()
    """
    def __init__(self, logger: logging.Logger, step_name: str, emoji: str = "⏱️"):
        self.logger = logger
        self.step_name = step_name
        self.emoji = emoji
        self._start: Optional[float] = None

    def __enter__(self):
        try:
            if self.logger:
                self.logger.info(f"{self.emoji} {self.step_name}...")
        finally:
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = 0.0 if self._start is None else (time.perf_counter() - self._start)
        if exc_type is not None:
            try:
                self.logger.error(f"❌ {self.step_name} failed after {duration:.2f}s: {exc}")
            except Exception:
                pass
            # Do not suppress exceptions
            return False
        try:
            self.logger.info(f"✅ {self.step_name} completed in {duration:.2f}s")
        except Exception:
            pass
        return False

@dataclass
class DateConversionResult:
    """Result of date conversion with detailed metadata."""
    converted_series: pd.Series
    success_rate: float
    failed_indices: List[int]
    warnings: List[str]
    recommendations: List[str]


class SmartDateHandler:
    """
    Advanced date handling with intelligent error recovery and validation.
    Provides robust date processing with comprehensive error analysis.
    """
    
    def __init__(self, profile):
        self.profile = profile
        self.date_columns = getattr(profile, 'date_columns', [])
        self.conversion_cache = {}  # Cache for expensive conversions
    
    def intelligent_date_conversion(self, series: pd.Series, column_name: str = None) -> DateConversionResult:
        """
        Intelligently convert series to datetime with multiple fallback strategies.
        Uses progressive conversion attempts with detailed error analysis.
        """
        original_series = series.copy()
        warnings = []
        recommendations = []
        
        # Strategy 1: Direct conversion with error tolerance
        try:
            converted = pd.to_datetime(series, errors='coerce')
            failed_mask = converted.isna() & original_series.notna()
            failed_count = failed_mask.sum()
            
            if failed_count == 0:
                return DateConversionResult(
                    converted_series=converted,
                    success_rate=1.0,
                    failed_indices=[],
                    warnings=[],
                    recommendations=["All dates converted successfully"]
                )
            
            # Strategy 2: Try alternative formats for failed values
            if failed_count > 0:
                converted = self._attempt_format_recovery(original_series, converted, failed_mask)
                failed_mask = converted.isna() & original_series.notna()
                failed_count = failed_mask.sum()
            
            # Strategy 3: Manual parsing for remaining failures
            if failed_count > 0:
                converted = self._manual_date_parsing(original_series, converted, failed_mask)
                failed_mask = converted.isna() & original_series.notna()
                failed_count = failed_mask.sum()
            
            success_rate = (len(series) - failed_count) / len(series) if len(series) > 0 else 0
            
            if failed_count > 0:
                warnings.append(f"{failed_count} values could not be converted to datetime")
                recommendations.append("Review data quality and consider manual cleanup")
            
            if success_rate < 0.8:
                warnings.append("Low conversion success rate detected")
                recommendations.append("Consider standardizing date format before processing")
            
            return DateConversionResult(
                converted_series=converted,
                success_rate=success_rate,
                failed_indices=failed_mask[failed_mask].index.tolist(),
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Date conversion failed for '{column_name}': {e}")
            return DateConversionResult(
                converted_series=original_series,
                success_rate=0.0,
                failed_indices=list(range(len(series))),
                warnings=[f"Conversion failed: {e}"],
                recommendations=["Check data format and try manual conversion"]
            )
    
    def _attempt_format_recovery(self, original: pd.Series, converted: pd.Series, failed_mask: pd.Series) -> pd.Series:
        """Attempt to recover failed conversions using alternative formats."""
        recovery_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m-%d-%Y',
            '%d-%m-%Y'
        ]
        
        result = converted.copy()
        failed_values = original[failed_mask]
        
        for fmt in recovery_formats:
            for idx in failed_values.index:
                try:
                    parsed_date = pd.to_datetime(failed_values.loc[idx], format=fmt)
                    if not pd.isna(parsed_date):
                        result.loc[idx] = parsed_date
                except:
                    continue
        
        return result
    
    def _manual_date_parsing(self, original: pd.Series, converted: pd.Series, failed_mask: pd.Series) -> pd.Series:
        """Manual parsing for complex date formats."""
        result = converted.copy()
        failed_values = original[failed_mask]
        
        for idx in failed_values.index:
            value = str(failed_values.loc[idx]).strip()
            
            # Handle common patterns
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}', value):
                try:
                    result.loc[idx] = pd.to_datetime(value, format='%Y-%m-%d')
                    continue
                except:
                    pass
            
            # Handle relative dates
            if 'today' in value.lower() or 'now' in value.lower():
                result.loc[idx] = pd.Timestamp.now()
                continue
            
            # Handle epoch timestamps
            if value.isdigit() and len(value) >= 10:
                try:
                    if len(value) == 10:  # Unix timestamp
                        result.loc[idx] = pd.to_datetime(int(value), unit='s')
                    elif len(value) == 13:  # Millisecond timestamp
                        result.loc[idx] = pd.to_datetime(int(value), unit='ms')
                    continue
                except:
                    pass
        
        return result
    
    def process_date_range_query(self, df: pd.DataFrame, column: str, 
                                start_value: Union[str, datetime], 
                                end_value: Union[str, datetime]) -> pd.DataFrame:
        """
        Process date range queries with intelligent parsing and validation.
        """
        if column not in df.columns:
            logger.warning(f"Date column '{column}' not found in DataFrame")
            return df
        
        # Parse date values
        start_date = self._parse_date_value(start_value)
        end_date = self._parse_date_value(end_value)
        
        if start_date is None or end_date is None:
            logger.error(f"Failed to parse date range: start='{start_value}', end='{end_value}'")
            return df
        
        # Ensure proper date order
        if start_date > end_date:
            start_date, end_date = end_date, start_date
            logger.info("Swapped start and end dates to ensure proper order")
        
        # Apply intelligent filtering
        return self._apply_date_range_filter(df, column, start_date, end_date)
    
    def _parse_date_value(self, value: Union[str, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """Parse various date value formats with fallback strategies."""
        if value is None:
            return None
        
        # Handle pandas Timestamp
        if isinstance(value, pd.Timestamp):
            return value
        
        # Handle datetime objects
        if isinstance(value, datetime):
            return pd.Timestamp(value)
        
        # Handle string values
        if isinstance(value, str):
            return self._parse_date_string(value)
        
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse date strings with multiple format support."""
        date_str = date_str.strip()
        
        # Try direct parsing first
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            pass
        
        # Try natural language parsing
        parsed = self._parse_natural_language_date(date_str)
        if parsed:
            return parsed
        
        # Try format-specific parsing
        return self._parse_with_known_formats(date_str)
    
    def _parse_natural_language_date(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse natural language date expressions."""
        date_str_lower = date_str.lower()
        now = pd.Timestamp.now()
        
        # Relative date patterns
        patterns = {
            r'today': now,
            r'yesterday': now - timedelta(days=1),
            r'tomorrow': now + timedelta(days=1),
            r'this week': now - timedelta(days=now.weekday()),
            r'last week': now - timedelta(days=7 + now.weekday()),
            r'this month': now.replace(day=1),
            r'last month': (now.replace(day=1) - timedelta(days=1)).replace(day=1),
            r'this year': now.replace(month=1, day=1),
            r'last year': now.replace(year=now.year-1, month=1, day=1),
        }
        
        for pattern, date_value in patterns.items():
            if re.search(pattern, date_str_lower):
                return date_value
        
        # Dynamic relative patterns
        dynamic_patterns = [
            (r'(\d+)\s+days?\s+ago', lambda m: now - timedelta(days=int(m.group(1)))),
            (r'(\d+)\s+weeks?\s+ago', lambda m: now - timedelta(weeks=int(m.group(1)))),
            (r'(\d+)\s+months?\s+ago', lambda m: now - timedelta(days=int(m.group(1)) * 30)),
            (r'(\d+)\s+years?\s+ago', lambda m: now - timedelta(days=int(m.group(1)) * 365)),
            (r'in\s+(\d+)\s+days?', lambda m: now + timedelta(days=int(m.group(1)))),
            (r'in\s+(\d+)\s+weeks?', lambda m: now + timedelta(weeks=int(m.group(1)))),
        ]
        
        for pattern, handler in dynamic_patterns:
            match = re.search(pattern, date_str_lower)
            if match:
                try:
                    return handler(match)
                except:
                    continue
        
        return None
    
    def _parse_with_known_formats(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse with known date formats."""
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%d.%m.%Y',
            '%m.%d.%Y'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        return None
    
    def _apply_date_range_filter(self, df: pd.DataFrame, column: str, 
                                start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Apply date range filter with intelligent column handling."""
        try:
            # Convert column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = pd.to_datetime(df[column], errors='coerce')
            
            # Apply filter
            mask = (df[column] >= start_date) & (df[column] <= end_date)
            filtered_df = df.loc[mask]
            
            # Log results
            original_count = len(df)
            filtered_count = len(filtered_df)
            logger.info(f"Date range filter: {filtered_count}/{original_count} rows match ({start_date.date()} to {end_date.date()})")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Date range filtering failed: {e}")
            return df


# Legacy function for backward compatibility
def parse_relative_date_range(text: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Parse relative date range expressions (legacy function for backward compatibility).
    
    Examples:
        - "last week" -> (start_of_last_week, end_of_last_week)
        - "last 30 days" -> (30_days_ago, now)
        - "past 2 months" -> (2_months_ago, now)
    """
    t = text.lower()
    now = cast(pd.Timestamp, pd.Timestamp.now().normalize())

    if re.search(r"last\s+week", t):
        end = now
        start = cast(pd.Timestamp, end - pd.Timedelta(days=7))
        return cast(Tuple[pd.Timestamp, pd.Timestamp], (start, end))
    if re.search(r"last\s+month", t):
        end = now
        start = cast(pd.Timestamp, end - pd.Timedelta(days=30))
        return cast(Tuple[pd.Timestamp, pd.Timestamp], (start, end))

    m = re.search(r"last\s+(\d+)\s+(day|days|week|weeks|month|months)", t)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        days = num
        if unit.startswith('week'):
            days = num * 7
        elif unit.startswith('month'):
            days = num * 30
        end = now
        start = cast(pd.Timestamp, end - pd.Timedelta(days=days))
        return cast(Tuple[pd.Timestamp, pd.Timestamp], (start, end))

    m2 = re.search(r"past\s+(\d+)\s+(day|days|week|weeks|month|months)", t)
    if m2:
        num = int(m2.group(1))
        unit = m2.group(2)
        days = num
        if unit.startswith('week'):
            days = num * 7
        elif unit.startswith('month'):
            days = num * 30
        end = now
        start = cast(pd.Timestamp, end - pd.Timedelta(days=days))
        return cast(Tuple[pd.Timestamp, pd.Timestamp], (start, end))
    return None