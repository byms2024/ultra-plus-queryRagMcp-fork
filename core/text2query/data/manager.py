"""Consolidated data management with enhanced cleaning and validation."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from config.base_config import Config
from config.profiles import DataProfile
from config.logging_config import get_rag_logger
from ..utils.time_utils import SmartDateHandler

logger = get_rag_logger()


class DataManager:
    """
    Consolidated data management with enhanced cleaning, validation, and date handling.
    Combines original data management with LangChain-style cleaning and smart date processing.
    """
    
    def __init__(self, config: Config, profile: DataProfile):
        self.config = config
        self.profile = profile
        self.date_handler = SmartDateHandler(profile)
        self.df: Optional[pd.DataFrame] = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load CSV data, validate columns, and apply enhanced data cleaning."""
        try:
            # Load CSV data
            csv_file = self.profile.get_data_file_path()
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Validate required columns
            missing = self.profile.validate_schema(df.columns.tolist())
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")
            
            # Apply enhanced data cleaning
            df = self._enhanced_clean_dataframe(df)
            
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to load and process data: {e}")
            raise
    
    def _enhanced_clean_dataframe(self, df: pd.DataFrame, apply_censoring: bool = True) -> pd.DataFrame:
        """
        Enhanced data cleaning that combines:
        1. LangChain-style cleaning
        2. Profile-specific cleaning rules
        3. Smart date handling
        4. Censoring system
        """
        logger.info(f"Starting enhanced data cleaning for {len(df)} rows")
        
        # Step 1: Apply LangChain-style cleaning
        df_cleaned = self._clean_dataframe_langchain_style(df)
        logger.debug("Applied LangChain-style cleaning")
        
        # Step 2: Apply profile-specific cleaning
        try:
            df_cleaned = self.profile.clean_data(df_cleaned)
            logger.debug("Applied profile-specific cleaning")
        except Exception as e:
            logger.warning(f"Profile cleaning failed, continuing with LangChain cleaning: {e}")
        
        # Step 3: Apply smart date processing
        df_cleaned = self._process_date_columns(df_cleaned)
        
        # Step 4: Apply censoring if requested
        if apply_censoring:
            try:
                df_cleaned = self._apply_profile_censoring(df_cleaned)
                logger.debug("Applied profile censoring")
            except Exception as e:
                logger.warning(f"Censoring failed, continuing without censoring: {e}")
        
        logger.info(f"Enhanced data cleaning completed. Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def _clean_dataframe_langchain_style(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the DataFrame using LangChain's approach:
        - Strip whitespace from string columns
        - Standardize empty strings as NaN
        - Convert date-like columns to datetime with error coercion
        """
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == object:
                # Strip whitespace and replace empty with NaN
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip().replace({"": None})
            
            # Attempt datetime conversion if column looks like dates
            if "date" in col.lower():
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors="coerce")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to datetime: {e}")
        
        return df_cleaned
    
    def _process_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process date columns with intelligent conversion."""
        for col in self.profile.date_columns:
            if col in df.columns:
                logger.info(f"Processing date column: {col}")
                
                # Perform intelligent conversion
                conversion_result = self.date_handler.intelligent_date_conversion(df[col], col)
                df[col] = conversion_result.converted_series
                
                # Log results
                if conversion_result.warnings:
                    for warning in conversion_result.warnings:
                        logger.warning(f"Date column '{col}': {warning}")
                
                if conversion_result.recommendations:
                    for rec in conversion_result.recommendations:
                        logger.info(f"Date column '{col}': {rec}")
        
        return df
    
    def _apply_profile_censoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply profile-specific censoring to sensitive columns.
        """
        try:
            censoring_mappings = self.profile.get_censoring_mappings()
            
            if not censoring_mappings:
                logger.debug("No censoring mappings found for profile")
                return df
            
            df_censored = df.copy()
            
            for column_name, censor_func in censoring_mappings.items():
                if column_name in df_censored.columns:
                    logger.debug(f"Applying censoring to column: {column_name}")
                    df_censored[column_name] = df_censored[column_name].apply(censor_func)
                else:
                    logger.debug(f"Censoring column '{column_name}' not found in DataFrame")
            
            return df_censored
            
        except Exception as e:
            logger.warning(f"Failed to apply censoring: {e}")
            return df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the loaded and processed DataFrame."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        return self.df
    
    def get_sample_data(self, n_rows: int = 3) -> str:
        """Get a sample of the data as JSON string for context."""
        if self.df is None:
            return ""
        
        try:
            # Convert timestamps to strings to avoid JSON serialization issues
            sample_df = self.df.head(n_rows).copy()
            for col in sample_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].astype(str)
            
            sample = sample_df.to_dict(orient="records")
            import json
            return json.dumps(sample, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to get sample data: {e}")
            return ""
    
    def apply_date_range_filter(self, df: pd.DataFrame, column: str, 
                               start_value: str, end_value: str) -> pd.DataFrame:
        """Apply date range filter using smart date processor."""
        return self.date_handler.process_date_range_query(df, column, start_value, end_value)


def enhanced_clean_dataframe(df: pd.DataFrame, profile: DataProfile, 
                           apply_censoring: bool = True) -> pd.DataFrame:
    """
    Standalone function for enhanced data cleaning (for backward compatibility).
    """
    # Create a temporary data manager for the cleaning operation
    class TempConfig:
        pass
    
    temp_config = TempConfig()
    temp_manager = DataManager(temp_config, profile)
    return temp_manager._enhanced_clean_dataframe(df, apply_censoring)


def validate_dataframe_for_langchain(df: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
    """
    Validate DataFrame for LangChain processing and return validation results.
    """
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "recommendations": [],
        "statistics": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    }
    
    # Check for missing required columns
    try:
        missing_columns = profile.validate_schema(df.columns.tolist())
        if missing_columns:
            validation_results["warnings"].append(f"Missing required columns: {missing_columns}")
            validation_results["recommendations"].append("Ensure all required columns are present")
    except Exception as e:
        validation_results["warnings"].append(f"Could not validate required columns: {e}")
    
    # Check for entirely null columns
    null_columns = []
    for col in df.columns:
        if df[col].isna().all():
            null_columns.append(col)
    
    if null_columns:
        validation_results["warnings"].extend([f"Column is entirely null: {col}" for col in null_columns])
        validation_results["recommendations"].append("Consider dropping entirely null columns")
    
    # Check for date columns with conversion issues
    date_columns = [col for col in df.columns if "date" in col.lower()]
    for col in date_columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='coerce')
            except Exception:
                validation_results["warnings"].append(f"Date column '{col}' may have format issues")
                validation_results["recommendations"].append(f"Review date format in column '{col}'")
    
    # Check for very wide DataFrames (might cause context issues)
    if len(df.columns) > 50:
        validation_results["warnings"].append(f"DataFrame has {len(df.columns)} columns, may cause context issues")
        validation_results["recommendations"].append("Consider selecting relevant columns for analysis")
    
    # Check for very long DataFrames (might cause performance issues)
    if len(df) > 10000:
        validation_results["warnings"].append(f"DataFrame has {len(df)} rows, may cause performance issues")
        validation_results["recommendations"].append("Consider sampling or filtering data for analysis")
    
    return validation_results


def build_schema_description(df: pd.DataFrame, profile: DataProfile) -> str:
    """
    Build a detailed schema description for the DataFrame.
    """
    schema_parts = []
    schema_parts.append("DataFrame columns and their inferred dtypes:")
    for col, dtype in df.dtypes.items():
        schema_parts.append(f"- {col}: {dtype}")
    
    # Add profile-specific schema hints
    try:
        sample_data = df.head(3).to_csv(index=False)
        profile_schema_hints = profile.get_schema_hints(sample_data)
        if profile_schema_hints:
            schema_parts.append("\nAdditional schema hints from profile:")
            schema_parts.append(profile_schema_hints)
    except Exception as e:
        logger.warning(f"Could not get profile schema hints: {e}")
    
    return "\n".join(schema_parts)