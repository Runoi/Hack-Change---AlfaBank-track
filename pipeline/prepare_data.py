import pandas as pd
import numpy as np
import os
import gc
import warnings
import sys

# --- CONFIG ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def clean_numeric_column(series):
    try:
        if series.dtype == 'object':
            cleaned = series.str.replace(',', '.', regex=False)
            return pd.to_numeric(cleaned, errors='coerce').astype('float32')
        else:
            return pd.to_numeric(series, errors='coerce').astype('float32')
    except Exception:
        return series 

def generate_features_simple(df):
    print("   [FE] Generating BASIC features...")
    
    if 'dt' in df.columns:
        dt_series = pd.to_datetime(df['dt'], errors='coerce')
        df['month'] = dt_series.dt.month.astype('float32')
        df['quarter'] = dt_series.dt.quarter.astype('float32')

    for col in df.columns:
        if '_12m' in col:
            base = col.replace('_12m', '')
            col_3m = base + '_3m'
            col_6m = base + '_6m'
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if col_3m in df.columns:
                    df[f'trend_3m_12m_{base}'] = (df[col_3m] + 1) / (df[col] + 1)
                if col_6m in df.columns:
                    df[f'trend_6m_12m_{base}'] = (df[col_6m] + 1) / (df[col] + 1)

    return df

def process_file(filepath, is_train=True):
    filename = os.path.basename(filepath)
    print(f"\n[INFO] Чтение файла: {filename}...")
    
    df = pd.read_csv(filepath, sep=';', engine='python')

    print("   [CLEAN] Исправление типов...")
    obj_cols = df.select_dtypes(include=['object']).columns
    skip_cols = [cfg.cols.id, 'dt']
    
    for col in obj_cols:
        if col in skip_cols: continue
        
        sample = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ""
        if any(c.isdigit() for c in sample):
            converted = clean_numeric_column(df[col])
            if converted.isna().sum() <= df[col].isna().sum() + len(df) * 0.5:
                 df[col] = converted

    df = generate_features_simple(df)
    
    if 'dt' in df.columns:
        df = df.drop(columns=['dt'])

    print("   [CLEAN] Simple Median Imputation...")
    
    num_cols = df.select_dtypes(include=['number']).columns
    protected_cols = [cfg.cols.id, cfg.cols.target, cfg.cols.weight]
    
    for col in num_cols:
        if col in protected_cols: continue
        
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
            
        df[col] = df[col].fillna(median_val).astype('float32')

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).replace('nan', 'Unknown').fillna("Unknown").astype("category")

    return df

def main():
    os.makedirs(cfg.paths.processed_data, exist_ok=True)

    train_raw_path = cfg.get_train_raw_path()
    if os.path.exists(train_raw_path):
        df_train = process_file(train_raw_path, is_train=True)
        df_train.to_parquet(cfg.get_train_proc_path(), index=False)
        print(f" Train сохранен")
        del df_train
        gc.collect()

    test_raw_path = cfg.get_test_raw_path()
    if os.path.exists(test_raw_path):
        df_test = process_file(test_raw_path, is_train=False)
        df_test.to_parquet(cfg.get_test_proc_path(), index=False)
        print(f" Test сохранен")

if __name__ == "__main__":
    main()