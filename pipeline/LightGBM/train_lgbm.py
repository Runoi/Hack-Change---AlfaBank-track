import json
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import gc
import os
import sys
import joblib
from sklearn.preprocessing import LabelEncoder

# --- МАГИЯ ИМПОРТА КОНФИГА ---
# Добавляем родительскую директорию в путь, чтобы видеть config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg
# -----------------------------

def wmae_metric(preds, train_data):
    """Кастомная метрика для LGBM"""
    labels = train_data.get_label()
    weights = train_data.get_weight()
    
    preds_arr = np.array(preds, dtype=float)
    labels_arr = np.array(labels, dtype=float)
    
    if weights is None:
        weights_arr = np.ones_like(labels_arr, dtype=float)
    else:
        weights_arr = np.array(weights, dtype=float)
    
    preds_real = np.expm1(preds_arr)
    labels_real = np.expm1(labels_arr)
    
    diff = np.abs(preds_real - labels_real)
    
    wmae = np.sum(diff * weights_arr) / np.sum(weights_arr)
    
    return 'wmae', wmae, False

def clean_float_col(series):
    if series.dtype == 'object' or series.dtype.name == 'category':
        return series.astype(str).str.replace(',', '.', regex=False).astype(float)
    return series

def load_data():
    print(f" Загрузка Parquet из {cfg.paths.processed_data}...")
    df_train = pd.read_parquet(cfg.get_train_proc_path())
    df_test = pd.read_parquet(cfg.get_test_proc_path())
    
    df_train[cfg.cols.target] = clean_float_col(df_train[cfg.cols.target])
    df_train[cfg.cols.weight] = clean_float_col(df_train[cfg.cols.weight])
    
    return df_train, df_test

def main():
    train_df, test_df = load_data()
    
    drop_cols = cfg.cols.drop.copy()
    
    if 'incomeValue' in test_df.columns:
        try:
            check_val = clean_float_col(test_df['incomeValue'])
            if check_val.sum() > 0:
                print(" ВНИМАНИЕ: 'incomeValue' найден в тесте! Используем как фичу.")
                if 'incomeValue' in drop_cols: drop_cols.remove('incomeValue')
        except:
            pass

    features = [c for c in train_df.columns if c not in drop_cols and c in test_df.columns]
    
    cat_cols = [c for c in features if train_df[c].dtype == 'object' or train_df[c].dtype.name == 'category']
    print(f" Кодирование категорий с помощью LabelEncoder ({len(cat_cols)} шт)...")
    label_encoders = {}
    
    for col in cat_cols:
        all_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        
        le = LabelEncoder()
        le.fit(all_values)
        
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        
        label_encoders[col] = le
    
    print(f" Приведение типов для категорий ({len(cat_cols)} шт)...")
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    X = train_df[features].copy().reset_index(drop=True)
    y_log = pd.Series(np.log1p(train_df[cfg.cols.target])).reset_index(drop=True)
    w = pd.Series(train_df[cfg.cols.weight]).reset_index(drop=True)
    
    X_test = test_df[features].copy()

    params = {
        'objective': 'regression_l1', 
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'learning_rate': cfg.lgbm_params.learning_rate,
        'num_leaves': cfg.lgbm_params.num_leaves,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'device': cfg.lgbm_params.device,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'seed': cfg.seed
    }

    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    scores = []

    

    print(f" Старт обучения LightGBM ({cfg.folds} фолдов)...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        y_tr = y_log.iloc[train_idx]
        y_val = y_log.iloc[val_idx]
        
        w_tr = w.iloc[train_idx]
        w_val = w.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg.lgbm_params.num_boost_round,
            valid_sets=[dtrain, dval],
            feval=wmae_metric, 
            callbacks=[
                lgb.early_stopping(stopping_rounds=cfg.lgbm_params.stopping_rounds),
                lgb.log_evaluation(period=500) 
            ]
        )

        val_pred_log = model.predict(X_val)
        val_pred_real = np.expm1(val_pred_log) # type: ignore
        
        oof_preds[val_idx] = val_pred_real
        
        diff = np.abs(np.expm1(y_val.values) - val_pred_real) # type: ignore
        weights_fold = w_val.values
        score = np.sum(diff * weights_fold) / np.sum(weights_fold) # type: ignore
        
        scores.append(score)
        print(f" Fold {fold+1} WMAE: {score:,.2f}")

        test_preds += np.expm1(model.predict(X_test)) / cfg.folds # type: ignore

        if fold == cfg.folds - 1:
            try:
                model_path = cfg.get_submission_path("model_lgbm.pkl")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                joblib.dump(model, model_path)
                    
                print(f" Модель для Docker сохранена (joblib): {model_path}")
            except Exception as e:
                print(f" Не удалось сохранить модель LightGBM: {e}")

        gc.collect()
        
    oof_path = cfg.get_submission_path("oof_lgbm.csv")
    oof_df = pd.DataFrame({
        cfg.cols.id: train_df[cfg.cols.id],
        'target': train_df[cfg.cols.target],
        'w': train_df[cfg.cols.weight],
        'predict': oof_preds
    })
    oof_df.to_csv(oof_path, index=False)

    print(f" OOF сохранен: {oof_path}")
    print("\n==============================")
    print(f" LGBM Avg WMAE: {np.mean(scores):,.2f}")
    
    diff_oof = np.abs(np.expm1(y_log.values) - oof_preds) # type: ignore
    oof_wmae = np.sum(diff_oof * w.values) / np.sum(w.values) # type: ignore
    print(f" OOF WMAE: {oof_wmae:,.2f}")
    print("==============================\n")

    sub_path = cfg.get_submission_path("submission_lgbm.csv")
    sub = pd.DataFrame({
        cfg.cols.id: test_df[cfg.cols.id],
        'predict': test_preds
    })
    
    sub.to_csv(sub_path, index=False)
    print(f" LGBM Сабмит сохранен: {sub_path}")

    features_path = cfg.get_submission_path("features.json")
    with open(features_path, 'w') as f:
        json.dump(features, f)
    print(f" Список фичей сохранен: {features_path}")

    print(" Сохранение артефактов для инференса...")
    artifacts = {
        'features': features,
        'cat_features': cat_cols,
        'label_encoders': label_encoders 
    }
    
    artifact_path = cfg.get_submission_path("lgbm_artifacts.pkl")
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print(f" Артефакты сохранены: {artifact_path}")

if __name__ == "__main__":
    main()