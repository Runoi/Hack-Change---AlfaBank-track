import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from config import cfg

PYTHON_EXE = sys.executable

def run_step(script_path, name):
    print(f"\n{'='*60}")
    print(f" ЗАПУСК: {name}")
    print(f" Файл: {script_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(script_path):
        print(f" Скрипт не найден: {script_path}")
        sys.exit(1)

    result = subprocess.run([PYTHON_EXE, script_path], check=False)
    
    if result.returncode != 0:
        print(f"\n ОШИБКА в модуле {name} (Код {result.returncode})")
        sys.exit(1)
    else:
        print(f"\n Модуль {name} успешно завершен.")

def wmae_metric(y_true, y_pred, weights):
    """Взвешенная MAE"""
    return np.sum(np.abs(y_true - y_pred) * weights) / np.sum(weights)

def optimize_and_blend():
    print(f"\n ЗАПУСК УМНОГО БЛЕНДИНГА (AutoML Mode)")
    print(f"{'-'*60}")
    
    models = ['lgbm', 'catboost', 'nn']
    
    oof_data = {}
    target = None
    w_col = None
    valid_models = []
    
    for m in models:
        path = cfg.get_submission_path(f"oof_{m}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if m == 'nn':
                df['predict'] = df['predict'].clip(upper=999_999) 
            oof_data[m] = np.log1p(df['predict'].values) # type: ignore
            valid_models.append(m)
            
            if target is None:
                target = df['target'].values
                w_col = df['w'].values
        else:
            print(f" OOF для {m} не найден. Пропускаем.")

    if not valid_models:
        print(" Нет данных для ансамбля.")
        return

    print(f" Оптимизация весов для: {valid_models}...")
    
    def loss_func(weights):
        final_log = np.zeros_like(oof_data[valid_models[0]])
        for i, m in enumerate(valid_models):
            final_log += oof_data[m] * weights[i]
        
        final_pred = np.expm1(final_log)
        
        return wmae_metric(target, final_pred, w_col)

    init_w = [1.0 / len(valid_models)] * len(valid_models)
    
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(valid_models)
    
    res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x
    best_score = res.fun
    
    print(f"\n ОПТИМИЗАЦИЯ ЗАВЕРШЕНА (Best OOF WMAE: {best_score:,.2f})")
    weights_dict = {}
    for m, w in zip(valid_models, best_weights):
        print(f"   -> {m.upper()}: {w:.4f}")
        weights_dict[m] = w

    print(f"\n Генерация финального сабмита...")
    
    final_test_log = 0
    base_id = None
    
    for m in valid_models:
        path = cfg.get_submission_path(f"submission_{m}.csv")
        df = pd.read_csv(path)

        
        if base_id is None:
            base_id = df[cfg.cols.id]
        
        if m == 'nn':
                df['predict'] = df['predict'].clip(upper=999_000) 
        pred_log = np.log1p(df['predict'].values) # type: ignore
        final_test_log += pred_log * weights_dict[m]
        
    final_pred = np.expm1(final_test_log)
    
    out_path = cfg.get_submission_path("submission_AUTO_ENSEMBLE.csv")
    pd.DataFrame({cfg.cols.id: base_id, 'target': final_pred}).to_csv(out_path, index=False)
    
    print(f"{'='*60}")
    print(f" ФАЙЛ ГОТОВ: {out_path}")
    print(f"{'='*60}\n")

def main():
    run_step("prepare_data.py", "Data Preparation")
    
    run_step("CatBoost/train_catboost.py", "CatBoost Training")
    run_step("LightGBM/train_lgbm.py", "LightGBM Training")
    run_step("NN/train_nn.py", "Neural Network Training")
    
    optimize_and_blend()

if __name__ == "__main__":
    main()