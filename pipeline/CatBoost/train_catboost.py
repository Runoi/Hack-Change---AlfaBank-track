import pickle
import warnings
import pandas as pd
import numpy as np
import os
import sys
import gc
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
warnings.filterwarnings("ignore", message=".*Failed to optimize method.*")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg

class WMAE_Metric(object):
    
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        preds_log = np.array(approxes[0])
        target_log = np.array(target)
        
        preds_real = np.expm1(preds_log)
        target_real = np.expm1(target_log)
        
        abs_error = np.abs(preds_real - target_real)
        
        if weight is None:
            weight = np.ones_like(abs_error)
        else:
            weight = np.array(weight)
            
        wmae = np.sum(abs_error * weight) / np.sum(weight)
        return wmae, 1

def clean_float_col(series):
    
    if series.dtype == 'object' or series.dtype.name == 'category':
        return series.astype(str).str.replace(',', '.', regex=False).astype(float)
    return series

def load_data():
    print(f" Загрузка Parquet из {cfg.paths.processed_data}...")
    df_train = pd.read_parquet(cfg.get_train_proc_path())
    df_test = pd.read_parquet(cfg.get_test_proc_path())
    
    print(" Конвертация таргета и весов...")
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
    
    cat_features = [c for c in features if train_df[c].dtype.name == 'category' or train_df[c].dtype == 'object']
    
    print(f" Фичей: {len(features)}. Категориальных: {len(cat_features)}")
    
    X = train_df[features].copy()
    y = train_df[cfg.cols.target].copy()
    w = train_df[cfg.cols.weight].copy()
    
    X_test = test_df[features].copy()
    
    print(" Подготовка категориальных фич...")
    for col in cat_features:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
    X = X.fillna(0)
    X_test = X_test.fillna(0)
    
    y_log = np.log1p(y)
    
    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    scores = []
    models = []
    
    print(f"\n Старт обучения CatBoost ({cfg.folds} фолдов)...")
    
    X = X.reset_index(drop=True)
    y_log = pd.Series(y_log).reset_index(drop=True)
    w = pd.Series(w).reset_index(drop=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        print(f"\n--- Fold {fold+1}/{cfg.folds} ---")
        
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr_log = y_log.iloc[train_idx]
        y_val_log = y_log.iloc[val_idx]
        w_tr = w.iloc[train_idx]
        w_val = w.iloc[val_idx]
        
        train_pool = Pool(X_tr, y_tr_log, weight=w_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val_log, weight=w_val, cat_features=cat_features)
        
        model = CatBoostRegressor(
            iterations=cfg.cat_params.iterations,
            learning_rate=cfg.cat_params.learning_rate,
            depth=cfg.cat_params.depth,
            loss_function=cfg.cat_params.loss_function,
            eval_metric=WMAE_Metric(),
            task_type=cfg.cat_params.task_type,
            early_stopping_rounds=cfg.cat_params.early_stopping,
            verbose=500,
            random_seed=cfg.seed
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        val_pred_log = model.predict(X_val)
        val_pred_real = np.expm1(val_pred_log)
        
        oof_preds[val_idx] = val_pred_real
        
        score = np.average(np.abs(np.expm1(y_val_log) - val_pred_real), weights=w_val.values)
        scores.append(score)
        print(f" Fold {fold+1} WMAE: {score:,.2f}")
        
        test_preds += np.expm1(model.predict(X_test)) / cfg.folds
        models.append(model)
        gc.collect()

    if models:
        model_path = cfg.get_submission_path("model_catboost.cbm")
        models[-1].save_model(model_path)
        print(f" Модель для Docker сохранена: {model_path}")
    print(" Сохранение артефактов...")
    artifacts = {
        'features': features,
        'cat_features': cat_features
    }
    artifact_path = cfg.get_submission_path("catboost_artifacts.pkl")
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifacts, f)
        
    oof_path = cfg.get_submission_path("oof_catboost.csv")
    oof_df = pd.DataFrame({
        cfg.cols.id: train_df[cfg.cols.id],
        'target': train_df[cfg.cols.target],
        'w': train_df[cfg.cols.weight],
        'predict': oof_preds
    })
    oof_df.to_csv(oof_path, index=False)
    print(f" OOF сохранен: {oof_path}")
    print("\n==============================")
    print(f" Average WMAE: {np.mean(scores):,.2f}")
    oof_wmae = np.average(np.abs(np.expm1(y_log) - oof_preds), weights=w.values)
    print(f" OOF WMAE: {oof_wmae:,.2f}")
    print("==============================\n")
    
    sub_path = cfg.get_submission_path("submission_catboost.csv")
    
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    
    submission = pd.DataFrame({
        cfg.cols.id: test_df[cfg.cols.id],
        'predict': test_preds
    })
    
    submission.to_csv(sub_path, index=False)
    print(f" Сабмит сохранен: {sub_path}")
    
    print("\nFeature Importance (Top 10):")
    imp = np.array(models[-1].get_feature_importance())
    indices = np.argsort(imp)[::-1]
    for i in range(min(10, len(features))):
        idx = indices[i]
        print(f"{features[idx]}: {imp[idx]:.2f}")

    

if __name__ == "__main__":
    main()