import pandas as pd
import numpy as np
from config import cfg

def calculate_metrics():
    print(" РАСЧЕТ БИЗНЕС-МЕТРИК (на OOF данных)")
    
    try:
        df = pd.read_csv(cfg.get_submission_path("oof_catboost.csv")) 
    except:
        print(" Нет файла OOF. Сначала обучи модели.")
        return

    y_true = df['target']
    y_pred = df['predict']

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    print(f"\n MAPE (Средняя ошибка в %): {mape:.2f}%")

    threshold = 0.20
    error_ratio = np.abs(y_true - y_pred) / (y_true + 1)
    hit_rate = np.mean(error_ratio <= threshold) * 100
    print(f" Hit Rate @ 20% (Точность попадания): {hit_rate:.2f}% клиентов")

    print("\n Ошибка по сегментам (MAPE):")
    df['segment'] = pd.qcut(df['target'], q=[0, 0.33, 0.66, 1.0], labels=['Low', 'Mid', 'High'])
    
    segment_stats = df.groupby('segment', observed=False).apply(
        lambda x: np.mean(np.abs((x['target'] - x['predict']) / (x['target'] + 1))) * 100
    )
    print(segment_stats)

    overestimation = np.mean(y_pred > y_true) * 100
    underestimation = np.mean(y_pred <= y_true) * 100
    
    print(f"\n Баланс рисков:")
    print(f"   - Переоценка (Риск дефолта): {overestimation:.1f}% случаев")
    print(f"   - Недооценка (Упущенная выгода): {underestimation:.1f}% случаев")
    
    spearman = df['target'].corr(df['predict'], method='spearman')
    print(f"\n Ранжирующая способность (Spearman Correlation): {spearman:.4f}")

if __name__ == "__main__":
    calculate_metrics()