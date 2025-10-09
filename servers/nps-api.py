from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
from datetime import datetime
import os
import uvicorn
import logging
import pandas as pd
import oracledb
import argparse
import json

from nps import calculate_nps
from database import (
    set_environment_mode,
    get_dealer_info_from_oracle,
    get_nps_data_from_sales_db,
    get_all_sales_code_info,
    get_nps_data,
    get_nps_data_from_latin_america,
    get_alt_nps_data,
    get_active_dealers,
    obtain_bonus_ranking,
    get_excluded_questionnaires_count,
    get_excluded_questionnaire_ids,
    get_specific_aftersales_questionnaire_data
)
from cache_manager import CacheManager
import numpy as np
import threading
import time


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = FastAPI(title="NPS API", version="2.0", default_response_class=ORJSONResponse)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

AUTH_USERS = {
    'sales_user': 'kQ@IF##7VY',
    'admin': 'admin_secure_pass_2024',
}

def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username or ""
    password = credentials.password or ""
    if username not in AUTH_USERS or AUTH_USERS[username] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


# Cache
CACHE_DISABLED = False
cache = CacheManager(cache_dir='nps_cache', default_ttl=3600)
EXCLUDED_QUESTIONNAIRES_COUNT: Optional[int] = None
_DEALER_INFO_CACHE: Optional[pd.DataFrame] = None
_DEALER_INFO_LAST_REFRESH: float = 0.0
_DEALER_INFO_TTL_SECONDS: int = 1800

REGIONS = {
    'BRAZIL': 'Brazil',
    'LATIN_AMERICA': 'LatinAmerica'
}

DATASOURCES = {
    'DMS': 'dms',
    'SALES_DB': 'sales_db',
    'SALES': 'sales_db'
}

DEPARTMENTS = {
    'AFTERSALES': 'aftersales',
    'SALES': 'sales'
}


def cache_get(key: str):
    global EXCLUDED_QUESTIONNAIRES_COUNT
    if CACHE_DISABLED:
        return None
    try:
        current_count = get_excluded_questionnaires_count()
        if EXCLUDED_QUESTIONNAIRES_COUNT is None:
            EXCLUDED_QUESTIONNAIRES_COUNT = current_count
        elif current_count != EXCLUDED_QUESTIONNAIRES_COUNT:
            cache.delete(key)
            EXCLUDED_QUESTIONNAIRES_COUNT = current_count
            return None
    except Exception as _e:
        logger.warning(f"Unable to verify EXCLUDED_QUESTIONNAIRES count for cache key {key}: {_e}")
    return cache.get(key)


def cache_set(key: str, value, timeout: Optional[int] = None):
    if CACHE_DISABLED:
        return
    if timeout:
        cache.set(key, value, ttl=timeout)
    else:
        cache.set(key, value)


def generate_cache_key(start_date, end_date, dealer_codes_param=None, data_type=None,
                       datasource=None, department=None, region_param=None,
                       period=None, is_cumulative=None, filters=None):
    key_data = f"nps_scores_{start_date}_{end_date}"
    params = [dealer_codes_param, data_type, datasource, department, region_param, period, is_cumulative]
    for param in params:
        if param is not None:
            key_data += f"_{param}"
    if filters:
        filters_str = json.dumps(filters, sort_keys=True)
        import hashlib
        key_data += f"_filters_{hashlib.md5(filters_str.encode()).hexdigest()[:8]}"
    import hashlib
    return hashlib.md5(key_data.encode()).hexdigest()


def parse_dealer_codes(dealer_codes_param: Optional[str]):
    if not dealer_codes_param:
        return None
    return dealer_codes_param.split(',')


def apply_dealer_filter(df: pd.DataFrame, dealer_codes, department: str, region_param: str):
    if not dealer_codes or department != 'aftersales' or region_param != 'Brazil':
        return df
    # Normalize once
    if 'DEALER_CODE' in df.columns:
        df['DEALER_CODE'] = df['DEALER_CODE'].astype(str).str.strip()
        df = df[df['DEALER_CODE'].isin(dealer_codes)]
    return df


def format_nps_response_data(result_df: pd.DataFrame):
    formatted_data = []
    for _, row in result_df.iterrows():
        try:
            dealer_code = row.get('dealer_code', row.get('DEALER_CODE', ''))
            if isinstance(dealer_code, pd.Series):
                dealer_code = dealer_code.values[0]
            dealer_code = str(dealer_code).strip()
            date_value = row['date']
            if isinstance(date_value, str):
                formatted_date = date_value
            else:
                formatted_date = date_value.strftime('%Y-%m-%d')
            response_item = {
                'date': formatted_date,
                'dealer_code': dealer_code,
                'dealer_group': str(row['dealer_group']).strip() if not pd.isna(row['dealer_group']) else 'Unknown',
                'region': str(row['region']),
                'dealer_name': str(row['dealer_name']).strip() if not pd.isna(row['dealer_name']) else 'Unknown',
                'NPS_SCORE': round(float(row['nps_score']), 2),
                'total': int(row['total']),
                'promoters': int(row['promoters']),
                'detractors': int(row['detractors'])
            }
            if 'neutrals' in row and not pd.isna(row['neutrals']):
                response_item['neutrals'] = int(row['neutrals'])
            if 'year' in row and not pd.isna(row['year']):
                response_item['year'] = int(row['year'])
            if 'period' in row and not pd.isna(row['period']):
                response_item['period'] = int(row['period'])
            formatted_data.append(response_item)
        except Exception:
            continue
    return formatted_data


def fetch_nps_data(datasource, department, region_param, questionnaire_ids=None, start_date=None, end_date=None, dealer_codes=None):
    normalized_datasource = DATASOURCES.get(str(datasource).upper(), datasource)
    if datasource == 'sales':
        normalized_datasource = 'sales_db'
    df_nps = None
    if region_param == REGIONS['BRAZIL']:
        if normalized_datasource == DATASOURCES['DMS']:
            df_nps = get_nps_data(questionnaire_ids, start_date, end_date, dealer_codes=dealer_codes)
        elif normalized_datasource == DATASOURCES['SALES_DB']:
            df_nps = get_nps_data_from_sales_db(questionnaire_ids=questionnaire_ids, start_date=start_date, end_date=end_date, dealer_codes=dealer_codes)
            if department == DEPARTMENTS['AFTERSALES']:
                df_nps['SCORE'] = df_nps['AFTERSALES_SCORE']
                df_nps = df_nps[df_nps['QUESTION_TYPE'].isin(['nps-1y-C', 'nps-3m-A', 'nps-2y-C'])]
            else:
                df_nps['SCORE'] = df_nps['SALES_SCORE']
                df_nps = df_nps[~df_nps['QUESTION_TYPE'].isin(['nps-1y-C', 'nps-2y-C', 'nps-1y-D'])]
    elif region_param == REGIONS['LATIN_AMERICA']:
        df_nps = get_nps_data_from_latin_america(questionnaire_ids=questionnaire_ids, start_date=start_date, end_date=end_date)
        if department == DEPARTMENTS['AFTERSALES']:
            df_nps = df_nps[df_nps['QUESTION_TYPE'].isin(['nps-1y-C', 'nps-3m-A', 'nps-2y-C'])]
            df_nps['SCORE'] = df_nps['AFTERSALES_SCORE']
        else:
            df_nps = df_nps[~df_nps['QUESTION_TYPE'].isin(['nps-1y-C', 'nps-2y-C', 'nps-1y-D'])]
            df_nps['SCORE'] = df_nps['SALES_SCORE']
    if df_nps is None:
        raise ValueError(f"No data source found for parameters: datasource={datasource} (normalized: {normalized_datasource}), region={region_param}, department={department}")
    return df_nps


def fetch_and_merge_dealer_info():
    global _DEALER_INFO_CACHE, _DEALER_INFO_LAST_REFRESH
    now = time.time()
    if _DEALER_INFO_CACHE is None or (now - _DEALER_INFO_LAST_REFRESH) > _DEALER_INFO_TTL_SECONDS:
        df_dealer_info = get_dealer_info_from_oracle()
        df_sales_code_info = get_all_sales_code_info()
        df_dealer_info = pd.merge(
            df_dealer_info, df_sales_code_info,
            left_on='DEALER CODE', right_on='DEALER_CODE',
            how='left'
        )
        if 'DEALER_CODE' in df_dealer_info.columns:
            df_dealer_info = df_dealer_info.drop('DEALER_CODE', axis=1)
        _DEALER_INFO_CACHE = df_dealer_info
        _DEALER_INFO_LAST_REFRESH = now
    return _DEALER_INFO_CACHE.copy()


def merge_nps_with_dealer_info(df_nps, df_dealer_info, department, region_param):
    if department == DEPARTMENTS['AFTERSALES'] and region_param == REGIONS['BRAZIL']:
        df_merged = pd.merge(
            df_nps, df_dealer_info, left_on='DEALER_CODE', right_on='DEALER CODE', how='left')
    elif department == DEPARTMENTS['SALES'] and region_param == REGIONS['BRAZIL']:
        df_nps_with_code = df_nps[df_nps['DEALER_CODE'].notna()].copy()
        df_nps_with_code = pd.merge(
            df_nps_with_code, df_dealer_info,
            left_on='DEALER_CODE', right_on='SALES_CODE',
            how='left'
        )
        df_nps_anonymous = df_nps[df_nps['DEALER_CODE'].isna()].copy()
        df_merged = pd.concat([df_nps_with_code, df_nps_anonymous], ignore_index=True)
    elif region_param == REGIONS['LATIN_AMERICA']:
        df_merged = pd.merge(
            df_nps, df_dealer_info, left_on='DEALER_CODE', right_on='DEALER CODE', how='left')
        df_merged = df_merged[df_merged['REGION'] == df_merged['Country']]
        df_merged = df_merged[df_merged['REGION NAME'] == df_merged['Country']]
    else:
        df_merged = df_nps
    return df_merged


def calculate_rolling_weekly_bins(df, extended_start_date, original_start_date, end_date):
    # Vectorized approach: daily aggregation per dealer, then sliding window using rolling on a 7D stride
    extended_start_date = pd.to_datetime(extended_start_date)
    original_start_date = pd.to_datetime(original_start_date)
    end_date = pd.to_datetime(end_date)
    df = df.copy()
    df['SUBMIT_DATE'] = pd.to_datetime(df['SUBMIT_DATE'])
    df = df[(df['SUBMIT_DATE'] >= extended_start_date) & (df['SUBMIT_DATE'] <= end_date)]
    # Categorize once
    df['category'] = pd.cut(df['SCORE'], bins=[-float('inf'), 6, 8, float('inf')], labels=['detractor', 'neutral', 'promoter'])
    # Normalize dealer code
    df['DEALER_CODE'] = df['DEALER_CODE'].astype(str).str.strip()
    # Daily counts per dealer
    daily = df.groupby(['DEALER_CODE', pd.Grouper(key='SUBMIT_DATE', freq='D')]).agg(
        total=('category', 'size'),
        promoters=('category', lambda s: (s == 'promoter').sum()),
        detractors=('category', lambda s: (s == 'detractor').sum()),
        neutrals=('category', lambda s: (s == 'neutral').sum()),
        dealer_group=('DEALER GROUP', lambda s: str(s.iloc[0]).strip() if len(s) else 'Unknown'),
        region_id=('REGION', lambda s: int(s.iloc[0]) if len(s) and not pd.isna(s.iloc[0]) else 0),
        region_name=('REGION NAME', lambda s: str(s.iloc[0]) if len(s) and not pd.isna(s.iloc[0]) else 'Unknown'),
        dealer_name=('STORE', lambda s: str(s.iloc[0]).strip() if len(s) and not pd.isna(s.iloc[0]) else 'Unknown'),
    ).reset_index().rename(columns={'SUBMIT_DATE': 'date'})
    # Compose region string
    daily['region'] = daily.apply(
        lambda row: f"{row['region_id']} - {row['region_name']}" if row['region_id'] != 0 and row['region_name'] != 'Unknown' else 'Unknown',
        axis=1
    )
    # Build weekly bin anchors (7-day steps)
    anchors = pd.date_range(start=original_start_date, end=end_date, freq='7D')
    results = []
    for i, anchor in enumerate(anchors, start=1):
        bin_start = max(extended_start_date, anchor - (original_start_date - extended_start_date))
        bin_end = anchor
        mask = (daily['date'] >= bin_start) & (daily['date'] <= bin_end)
        window = daily.loc[mask]
        if window.empty:
            continue
        agg = window.groupby('DEALER_CODE').agg(
            total=('total', 'sum'),
            promoters=('promoters', 'sum'),
            detractors=('detractors', 'sum'),
            neutrals=('neutrals', 'sum'),
            dealer_group=('dealer_group', 'first'),
            region=('region', 'first'),
            dealer_name=('dealer_name', 'first'),
        ).reset_index().rename(columns={'DEALER_CODE': 'dealer_code'})
        agg['bin_start'] = bin_start.date()
        agg['bin_end'] = bin_end.date()
        agg['bin_number'] = i
        agg['nps_score'] = ((agg['promoters'] - agg['detractors']) / agg['total'].replace(0, pd.NA)) * 100
        results.append(agg)
    if not results:
        return pd.DataFrame()
    result_df = pd.concat(results, ignore_index=True)
    return result_df


def format_rolling_weekly_response_data(result_df):
    formatted_data = []
    for _, row in result_df.iterrows():
        try:
            dealer_code = str(row['dealer_code']).strip()
            response_item = {
                'dealer_code': dealer_code,
                'bin_number': int(row['bin_number']),
                'bin_start': str(row['bin_start']),
                'bin_end': str(row['bin_end']),
                'dealer_group': str(row['dealer_group']).strip() if not pd.isna(row['dealer_group']) else 'Unknown',
                'region': str(row['region']),
                'dealer_name': str(row['dealer_name']).strip() if not pd.isna(row['dealer_name']) else 'Unknown',
                'nps_score': round(float(row['nps_score']), 2),
                'total': int(row['total']),
                'promoters': int(row['promoters']),
                'detractors': int(row['detractors']),
                'neutrals': int(row['neutrals'])
            }
            formatted_data.append(response_item)
        except Exception:
            continue
    return formatted_data

@app.on_event("startup")
def startup_event():
    # Parse CLI args if run directly with `python fastapi_app.py`
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, choices=['local', 'uat'], default='local')
    parser.add_argument('--no-cache', action='store_true')
    try:
        args, _ = parser.parse_known_args(sys.argv[1:])
    except SystemExit:
        class Dummy: pass
        args = Dummy()
        args.mode = 'local'
        args.no_cache = False

    # Allow environment variables to override flags when running under uvicorn
    env_mode = os.getenv('NPS_MODE')
    env_no_cache = os.getenv('NPS_NO_CACHE')

    mode = env_mode if env_mode in {'local', 'uat'} else args.mode
    no_cache = (str(env_no_cache).lower() in {'1', 'true', 'yes'}) if env_no_cache is not None else args.no_cache

    global CACHE_DISABLED
    CACHE_DISABLED = no_cache
    set_environment_mode(mode)
    try:
        oracledb.init_oracle_client()
    except Exception:
        # Already initialized or not required in this environment
        pass

    # Background refresher for dealer info
    def _refresh_dealer_info_loop():
        global _DEALER_INFO_CACHE, _DEALER_INFO_LAST_REFRESH
        while True:
            try:
                df = get_dealer_info_from_oracle()
                df_sales = get_all_sales_code_info()
                df = pd.merge(df, df_sales, left_on='DEALER CODE', right_on='DEALER_CODE', how='left')
                if 'DEALER_CODE' in df.columns:
                    df = df.drop('DEALER_CODE', axis=1)
                _DEALER_INFO_CACHE = df
                _DEALER_INFO_LAST_REFRESH = time.time()
            except Exception:
                pass
            time.sleep(_DEALER_INFO_TTL_SECONDS)

    threading.Thread(target=_refresh_dealer_info_loop, daemon=True).start()


@app.get("/nps/api/nps-scores")
def get_nps_scores(start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   dealer_codes: Optional[str] = None,
                   datasource: str = DATASOURCES['DMS'],
                   department: str = DEPARTMENTS['AFTERSALES']):
    region_param = REGIONS['BRAZIL']
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    cache_key = generate_cache_key(start_date, end_date, dealer_codes, 'scores', datasource, department, region_param)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    df_nps = fetch_nps_data(datasource=datasource, department=department, region_param=region_param, start_date=start_date, end_date=end_date, dealer_codes=dealer_codes_list)
    df_dealer_info = fetch_and_merge_dealer_info()
    df_merged = merge_nps_with_dealer_info(df_nps, df_dealer_info, department, region_param)
    # SQL-level filter already applied; no extra dealer filtering needed
    df_filtered = df_merged
    result_df = calculate_nps(df_filtered, 'daily')
    formatted = format_nps_response_data(result_df)
    cache_set(cache_key, formatted)
    return formatted


@app.get("/nps/api/nps-scores-country")
def get_nps_scores_country(start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           dealer_codes: Optional[str] = None,
                           datasource: str = DATASOURCES['DMS'],
                           department: str = DEPARTMENTS['AFTERSALES']):
    region_param = REGIONS['LATIN_AMERICA']
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    cache_key = generate_cache_key(start_date, end_date, dealer_codes, 'scores', datasource, department, region_param)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    df_nps = fetch_nps_data(datasource=datasource, department=department, region_param=region_param, start_date=start_date, end_date=end_date, dealer_codes=dealer_codes_list)
    df_nps['REGION'] = 1
    df_nps['REGION NAME'] = df_nps['Country']
    df_nps['STORE'] = df_nps['DEALER_CODE']
    df_nps['DEALER GROUP'] = df_nps['Country']
    # Common normalization
    if 'DEALER_CODE' in df_nps.columns:
        df_nps['DEALER_CODE'] = df_nps['DEALER_CODE'].astype(str).str.strip()
    df_filtered = df_nps.dropna(subset=['SCORE'])
    result_df = calculate_nps(df_filtered, 'daily')
    formatted = format_nps_response_data(result_df)
    cache_set(cache_key, formatted)
    return formatted


@app.get("/nps/api/nps-scores-rolling-weekly")
def get_nps_scores_rolling_weekly(start_date: str,
                                  end_date: str,
                                  dealer_codes: Optional[str] = None,
                                  datasource: str = DATASOURCES['DMS'],
                                  department: str = DEPARTMENTS['AFTERSALES']):
    region_param = REGIONS['BRAZIL']
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    n_days = 90
    extended_start_date = start_dt - pd.Timedelta(days=n_days)
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    df_nps = fetch_nps_data(datasource=datasource, department=department, region_param=region_param, start_date=start_dt, end_date=end_dt, dealer_codes=dealer_codes_list)
    df_dealer_info = fetch_and_merge_dealer_info()
    df_merged = merge_nps_with_dealer_info(df_nps, df_dealer_info, department, region_param)
    # SQL-level filter already applied; normalize for safety
    if 'DEALER_CODE' in df_merged.columns:
        df_merged['DEALER_CODE'] = df_merged['DEALER_CODE'].astype(str).str.strip()
    df_filtered = df_merged
    result_df = calculate_rolling_weekly_bins(df_filtered, extended_start_date, start_dt, end_dt)
    formatted = format_rolling_weekly_response_data(result_df)
    return formatted


# ======================= Questionnaires and Bonus Endpoints =======================
from urllib.parse import unquote_plus


def apply_questionnaire_filters(df: pd.DataFrame, dealer_codes, department: str):
    if (dealer_codes and len(dealer_codes) > 0 and 'DEALER_CODE' in df.columns and department == DEPARTMENTS['AFTERSALES']):
        df = df[df['DEALER_CODE'].isin(dealer_codes)]
    return df


def apply_date_filters(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]):
    if start_date:
        start_dt = pd.to_datetime(start_date, errors='coerce')
        if pd.notna(start_dt):
            df = df[df['SUBMIT_DATE'] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, errors='coerce')
        if pd.notna(end_dt):
            df = df[df['SUBMIT_DATE'] < end_dt]
    return df


def format_questionnaire_response_data(df: pd.DataFrame):
    formatted_data = []
    for _, row in df.iterrows():
        try:
            dealer_code = str(row.get('DEALER_CODE', '')).strip()
            if isinstance(dealer_code, pd.Series):
                dealer_code = dealer_code.values[0]
            questionnaire_data = {
                'questionnaire_number': str(row.get('ID', '')),
                'questionnaire_submission_date': (row['SUBMIT_DATE'].strftime('%Y-%m-%d') if pd.notna(row.get('SUBMIT_DATE')) else None),
                'score': int(row['SCORE']) if pd.notna(row.get('SCORE')) else None,
                'customer_type': str(row.get('category')) if pd.notna(row.get('category')) else None,
                'dealer_code': dealer_code,
                'dealer_name': str(row.get('STORE', '')).strip() if pd.notna(row.get('STORE')) else 'Unknown',
                'dealer_group': str(row.get('DEALER GROUP', '')).strip() if pd.notna(row.get('DEALER GROUP')) else 'Unknown',
                'region': str(row['REGION']) if pd.notna(row.get('REGION')) else 'Unknown',
            }
            string_fields = [
                ('ro_no', 'RO_NO'), ('vin', 'VIN'), ('question_three_reason', 'QUE_THREE_REASON'),
                ('repair_type', 'REPAIR_TYPE_NAME'), ('trouble_description', 'TROUBLE_DESC'),
                ('check_result', 'CHECK_RESULT'), ('deliver_problem', 'DELIVER_PROBLEM'),
                ('service_attitude', 'SERVICE_ATTITUDE'), ('environment', 'ENVIRONMENT'),
                ('efficiency', 'EFFICIENCY'), ('effectiveness', 'EFFECTIVENESS'),
                ('parts_availability', 'PARTS_AVAILABILITY'), ('others', 'OTHERS'),
                ('question_two_reason', 'QUE_TWO_REASON'), ('vehicle_model', 'VEHICLE_MODEL'),
                ('is_anonymous', 'IS_ANON'), ('completion_evidence', 'COMPLETION_EVIDENCE'),
                ('clng_status', 'CLNG_STATUS'), ('purchase_date', 'PURCHASE_DATE'),
                ('delivery_date', 'DELIVERY_DATE')
            ]
            for field, column in string_fields:
                questionnaire_data[field] = (str(row[column]) if column in row and pd.notna(row[column]) else None)
            date_fields = [
                ('order_create_date', 'ORDER_CREATE_DATE'), ('order_last_balance_date', 'ORDER_LAST_BALANCE_DATE'),
                ('action_date', 'ACTION_DATE'), ('close_date', 'CLOSE_DATE'),
                ('disputed_at', 'DISPUTED_AT'), ('reviewed_at', 'REVIEWED_AT')
            ]
            for field, column in date_fields:
                if column in row and pd.notna(row[column]):
                    if isinstance(row[column], (pd.Timestamp, datetime)):
                        questionnaire_data[field] = row[column].strftime('%Y-%m-%d')
                    else:
                        try:
                            dt = pd.to_datetime(row[column])
                            questionnaire_data[field] = dt.strftime('%Y-%m-%d')
                        except Exception:
                            questionnaire_data[field] = None
                else:
                    questionnaire_data[field] = None
            formatted_data.append(questionnaire_data)
        except Exception:
            continue
    return formatted_data


@app.get("/nps/api/contested-questionnaires")
def get_contested_questionnaires(start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 dealer_codes: Optional[str] = None,
                                 datasource: str = DATASOURCES['DMS'],
                                 department: str = DEPARTMENTS['AFTERSALES']):
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    df_dealer_info = fetch_and_merge_dealer_info()
    contested_ids = get_excluded_questionnaire_ids()
    df_quest = get_specific_aftersales_questionnaire_data(contested_ids)
    df_merged = merge_nps_with_dealer_info(df_quest, df_dealer_info, department, REGIONS['BRAZIL'])
    df_merged = apply_questionnaire_filters(df_merged, dealer_codes_list, department)

    def _to_jsonable(v):
        try:
            if v is None or pd.isna(v):
                return None
        except Exception:
            pass
        try:
            if isinstance(v, (pd.Timestamp, datetime)):
                return v.strftime('%Y-%m-%d')
            if isinstance(v, (np.datetime64,)):
                dt = pd.to_datetime(v, errors='coerce')
                return dt.strftime('%Y-%m-%d') if pd.notna(dt) else None
        except Exception:
            pass
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (int, float, str, bool)):
            return v
        try:
            return str(v)
        except Exception:
            return None

    records = []
    for _, row in df_merged.iterrows():
        rec = {}
        for col in df_merged.columns:
            try:
                rec[col] = _to_jsonable(row[col]) if col in row else None
            except Exception:
                rec[col] = None
        records.append(rec)
    return records


@app.get("/nps/api/bonus-ranking")
def get_bonus_ranking(dealer_codes: Optional[str] = None,
                      groups_param: Optional[str] = None):
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    df = obtain_bonus_ranking()
    if dealer_codes:
        mask_allowed = df['DEALER_CODE'].astype(str).isin(dealer_codes_list)
        if 'DEALER_NAME' in df.columns:
            df.loc[~mask_allowed, 'DEALER_NAME'] = '**********'
        if 'GROUP_NAME' in df.columns and not groups_param:
            df.loc[~mask_allowed, 'GROUP_NAME'] = '**********'
    if groups_param and 'GROUP_NAME' in df.columns:
        allowed_groups = set([unquote_plus(g).strip() for g in groups_param.split(',') if g and g.strip()])
        df['GROUP_NAME'] = df['GROUP_NAME'].astype(str).str.strip()
        df.loc[~df['GROUP_NAME'].isin(allowed_groups), 'GROUP_NAME'] = '**********'
    df.drop(axis=1, columns=['DEALER_CODE'], inplace=True)
    return df.to_dict(orient='records')


@app.get("/nps/api/bonus-ranking-group")
def get_bonus_ranking_group(groups_param: Optional[str] = None):
    if groups_param:
        allowed_groups = set([unquote_plus(g).strip() for g in groups_param.split(',') if g and g.strip()])
    else:
        allowed_groups = set()
    df = obtain_bonus_ranking()
    if 'GROUP_NAME' not in df.columns or 'DEALER_CODE' not in df.columns:
        raise HTTPException(status_code=500, detail='Required columns missing from bonus ranking data')
    df['DEALER_CODE'] = df['DEALER_CODE'].astype(str).str.strip()
    df['GROUP_NAME'] = df['GROUP_NAME'].astype(str).str.strip()
    if 'FINAL_BONUS' not in df.columns:
        raise HTTPException(status_code=500, detail='FINAL_BONUS column missing from bonus ranking data')
    group_df = (
        df.groupby(['GROUP_NAME', 'QUARTER', 'YEAR'], as_index=False)['FINAL_BONUS'].sum()
    )
    group_df = group_df.sort_values('FINAL_BONUS', ascending=False).reset_index(drop=True)
    group_df['RANKING'] = np.arange(1, len(group_df) + 1)
    group_df['GROUP_NAME'] = group_df['GROUP_NAME'].astype(str).str.strip()
    group_df.loc[~group_df['GROUP_NAME'].isin(allowed_groups), 'GROUP_NAME'] = '**********'
    return group_df.to_dict(orient='records')


@app.get("/nps/api/get-questionnaire")
def get_questionnaire(questionnaire_id: Optional[str] = None,
                      datasource: str = DATASOURCES['DMS'],
                      department: str = DEPARTMENTS['AFTERSALES']):
    df_nps = fetch_nps_data(
        datasource=datasource,
        department=department,
        region_param=REGIONS['BRAZIL'],
        questionnaire_ids=[questionnaire_id] if questionnaire_id else None
    )
    df_dealer_info = fetch_and_merge_dealer_info()
    df_merged = merge_nps_with_dealer_info(df_nps, df_dealer_info, department, REGIONS['BRAZIL'])
    # Normalize and coerce datetime columns directly on DataFrame
    date_columns = ['SUBMIT_DATE', 'ORDER_CREATE_DATE', 'ORDER_LAST_BALANCE_DATE', 'ACTION_DATE', 'CLOSE_DATE']
    for date_col in date_columns:
        if date_col in df_merged.columns:
            df_merged[date_col] = pd.to_datetime(df_merged[date_col], errors='coerce')
    df_final = df_merged[df_merged['ID'] == questionnaire_id]
    return format_questionnaire_response_data(df_final)


@app.get("/nps/api/questionnaires")
def get_questionnaires(start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       dealer_codes: Optional[str] = None,
                       datasource: str = DATASOURCES['DMS'],
                       department: str = DEPARTMENTS['AFTERSALES']):
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    df_nps = fetch_nps_data(datasource=datasource, department=department, region_param=REGIONS['BRAZIL'], start_date=start_date, end_date=end_date, dealer_codes=dealer_codes_list)
    df_dealer_info = fetch_and_merge_dealer_info()
    df_merged = merge_nps_with_dealer_info(df_nps=df_nps, df_dealer_info=df_dealer_info, department=department, region_param=REGIONS['BRAZIL'])
    # Coerce datetime columns once
    date_columns = ['SUBMIT_DATE', 'ORDER_CREATE_DATE', 'ORDER_LAST_BALANCE_DATE', 'ACTION_DATE', 'CLOSE_DATE']
    for date_col in date_columns:
        if date_col in df_merged.columns:
            df_merged[date_col] = pd.to_datetime(df_merged[date_col], errors='coerce')
    df_final = apply_questionnaire_filters(df=df_merged, dealer_codes=dealer_codes_list, department=department)
    if 'SCORE' in df_final.columns:
        df_final['category'] = pd.cut(df_final['SCORE'], bins=[-float('inf'), 6, 8, float('inf')], labels=['Detractor', 'Neutral', 'Promoter'])
    mask = (~df_final['RO_NO'].isnull()) & (df_final['REPAIR_TYPE_NAME'].isnull() | (df_final['REPAIR_TYPE_NAME'] == ''))
    df_final.loc[mask, 'REPAIR_TYPE_NAME'] = 'Order Unavailable in DMS'
    return format_questionnaire_response_data(df_final)


@app.get("/nps/api/questionnaire_category")
def get_questionnaire_category(dealer_codes: Optional[str] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None):
    dealer_codes_list = parse_dealer_codes(dealer_codes)
    df_nps = fetch_nps_data(datasource=DATASOURCES['DMS'], department=DEPARTMENTS['AFTERSALES'], region_param=REGIONS['BRAZIL'], start_date=start_date, end_date=end_date, dealer_codes=dealer_codes_list)
    df_dealer_info = fetch_and_merge_dealer_info()
    df_merged = merge_nps_with_dealer_info(df_nps, df_dealer_info, DEPARTMENTS['AFTERSALES'], REGIONS['BRAZIL'])
    df = df_merged
    if start_date or end_date:
        df = apply_date_filters(df, start_date, end_date)
    if dealer_codes_list:
        df['DEALER_CODE'] = df['DEALER_CODE'].astype(str).str.strip()
        df = df[df['DEALER_CODE'].isin(dealer_codes_list)]
    if df is None or len(df) == 0:
        return []
    categories = ['SERVICE_ATTITUDE', 'ENVIRONMENT', 'EFFICIENCY', 'EFFECTIVENESS']
    _df = df.loc[:, ~df.columns.duplicated()].copy()
    for col in categories:
        _df[col] = _df[col].astype(str).str.upper().eq('Y')
    _df['SCORE'] = pd.to_numeric(_df['SCORE'], errors='coerce')
    dealers = (_df[['DEALER_CODE']].drop_duplicates().sort_values('DEALER_CODE').reset_index(drop=True))
    result = dealers.set_index('DEALER_CODE')
    total_items = _df.groupby('DEALER_CODE').size().rename('total_items')
    result = result.join(total_items, how='left')
    for c in categories:
        mask_y = _df[c]
        prom = (_df.loc[mask_y & _df['SCORE'].between(9, 10, inclusive='both')].groupby('DEALER_CODE')['SCORE'].size().rename(f'{c}_promoters_n'))
        det = (_df.loc[mask_y & _df['SCORE'].between(0, 8, inclusive='both')].groupby('DEALER_CODE')['SCORE'].size().rename(f'{c}_detractors_n'))
        result = result.join(prom, how='left').join(det, how='left')
    count_cols = [col for col in result.columns if col.endswith('_promoters_n') or col.endswith('_detractors_n')]
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    for c in categories:
        prom_col = f'{c}_promoters_n'
        det_col = f'{c}_detractors_n'
        total_col = f'{c}_total'
        pct_col = f'{c}_pct'
        result[total_col] = result[prom_col] + result[det_col]
        total = result[total_col].replace(0, np.nan)
        result[pct_col] = np.round(((result[prom_col] - result[det_col]) / total) * 100, 2)
    result = result.reset_index()
    result_df = result.rename(columns={'DEALER_CODE': 'dealer_code'})
    try:
        dealer_info = get_dealer_info_from_oracle().rename(columns={'DEALER CODE': 'dealer_code'})
        result_df = result_df.merge(dealer_info[['dealer_code', 'STORE']], on='dealer_code', how='left')
    except Exception:
        pass
    result_df = result_df.replace({np.nan: None})
    return result_df.to_dict(orient='records')


def format_alt_nps_data(df: pd.DataFrame):
    df = df.rename(columns={
        'REPAIR_TYPE_NAME': 'repair_type',
        'SUBMIT_DATE': 'date',
        'SCORE': 'nps_score',
        'ID': 'questionnaire_id',
        'VIN': 'vin',
        'DEALER_CODE': 'dealer_code'
    })
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    formatted_data = []
    for _, row in df.iterrows():
        try:
            row_dict = {}
            for column in df.columns:
                value = row[column]
                if isinstance(value, pd.Series):
                    value = value.iloc[0] if len(value) > 0 else None
                if pd.isna(value) or value is None:
                    row_dict[column] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    row_dict[column] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, (int, float)):
                    row_dict[column] = value
                else:
                    row_dict[column] = str(value)
            formatted_data.append(row_dict)
        except Exception:
            continue
    return formatted_data


@app.get("/nps/api/aftersales-alt-nps")
def get_aftersales_alt_nps(start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           type: Optional[str] = 'national',
                           dealer_codes: Optional[str] = None,
                           group: Optional[str] = None,
                           region: Optional[str] = None,
                           username: str = Depends(verify_basic_auth)):
    request_type = type or 'national'
    allowed_types = {'national', 'regional', 'dealer', 'group'}
    if request_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid request_type: {request_type}. Allowed values are: {', '.join(allowed_types)}")
    df_nps = get_alt_nps_data()
    if not dealer_codes:
        dealer_codes_list = get_active_dealers()
    else:
        dealer_codes_list = dealer_codes.split(',')
    df_filtered = df_nps[df_nps['DEALER_CODE'].isin(dealer_codes_list)]
    df_filtered['TYPE'] = pd.cut(df_filtered['SCORE'], bins=[-float('inf'), 6, 8, float('inf')], labels=['DETRACTOR', 'NEUTRAL', 'PROMOTER'])
    if request_type == 'national':
        counts = df_filtered['TYPE'].value_counts()
        promoter = int(counts.get('PROMOTER', 0))
        detractor = int(counts.get('DETRACTOR', 0))
        neutral = int(counts.get('NEUTRAL', 0))
        nps_score = ((promoter - detractor) / max(promoter + neutral + detractor, 1)) * 100
        nps_score = round(nps_score, 1)
        return {'nps_score': nps_score, 'date_range': f"{start_date} to {end_date}", 'promoter': promoter, 'detractor': detractor, 'neutral': neutral}
    if request_type == 'regional':
        if region:
            df_filtered = df_filtered[df_filtered['REGION_NAME'].str.lower() == region.lower()]
        order_type_counts = df_filtered.groupby(['REGION_NAME'])['REPAIR_TYPE_NAME'].value_counts().to_dict()
        df_filtered = df_filtered.groupby(['REGION_NAME'])['TYPE'].value_counts().unstack(fill_value=0).reset_index().rename_axis(None, axis=1)
        df_filtered['SCORE'] = ((df_filtered['PROMOTER'] - df_filtered['DETRACTOR']) / (df_filtered['PROMOTER'] + df_filtered['NEUTRAL'] + df_filtered['DETRACTOR']) * 100).round(1)
        df_filtered = df_filtered[['REGION_NAME', 'SCORE', 'PROMOTER', 'DETRACTOR', 'NEUTRAL']].rename(columns={'REGION_NAME': 'region', 'PROMOTER': 'promoter', 'DETRACTOR': 'detractor', 'NEUTRAL': 'neutral', 'SCORE': 'nps_score'}).to_dict(orient='records')
        return df_filtered
    if request_type == 'dealer':
        df_filtered = df_filtered.groupby(['DEALER_CODE'])['TYPE'].value_counts().unstack(fill_value=0).reset_index().rename_axis(None, axis=1)
        df_filtered['SCORE'] = ((df_filtered['PROMOTER'] - df_filtered['DETRACTOR']) / (df_filtered['PROMOTER'] + df_filtered['NEUTRAL'] + df_filtered['DETRACTOR']) * 100).round(1)
        df_filtered = df_filtered[['DEALER_CODE', 'SCORE', 'PROMOTER', 'DETRACTOR', 'NEUTRAL']].rename(columns={'DEALER_CODE': 'dealer_code', 'PROMOTER': 'promoter', 'DETRACTOR': 'detractor', 'NEUTRAL': 'neutral', 'SCORE': 'nps_score'}).to_dict(orient='records')
        return df_filtered
    if request_type == 'group':
        if group:
            df_filtered = df_filtered[df_filtered['GROUP_NAME'].str.lower() == group.lower()]
        df_filtered = df_filtered.groupby(['GROUP_NAME'])['TYPE'].value_counts().unstack(fill_value=0).reset_index().rename_axis(None, axis=1)
        df_filtered['SCORE'] = ((df_filtered['PROMOTER'] - df_filtered['DETRACTOR']) / (df_filtered['PROMOTER'] + df_filtered['NEUTRAL'] + df_filtered['DETRACTOR']) * 100).round(1)
        df_filtered = df_filtered[['GROUP_NAME', 'SCORE', 'PROMOTER', 'DETRACTOR', 'NEUTRAL']].rename(columns={'GROUP_NAME': 'group', 'PROMOTER': 'promoter', 'DETRACTOR': 'detractor', 'NEUTRAL': 'neutral', 'SCORE': 'nps_score'}).to_dict(orient='records')
        return df_filtered
    formatted_data = format_alt_nps_data(df_filtered)
    return formatted_data

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPS FastAPI Server')
    parser.add_argument('--mode', type=str, choices=['local', 'uat'], default='local')
    parser.add_argument('--port', type=int, default=7777)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    env_mode = os.getenv('NPS_MODE')
    env_no_cache = os.getenv('NPS_NO_CACHE')
    mode = env_mode if env_mode in {'local', 'uat'} else args.mode
    no_cache = (str(env_no_cache).lower() in {'1', 'true', 'yes'}) if env_no_cache is not None else args.no_cache
    CACHE_DISABLED = no_cache
    set_environment_mode(mode)
    try:
        oracledb.init_oracle_client()
    except Exception:
        pass

    uvicorn.run("nps-api:app", host="0.0.0.0", port=args.port, reload=False)


