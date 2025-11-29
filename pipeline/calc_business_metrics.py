import pandas as pd
import numpy as np
from config import cfg

def calculate_metrics():
    print("📊 РАСЧЕТ БИЗНЕС-МЕТРИК (на OOF данных)")
    
    # Загружаем OOF ансамбля (или любой другой модели, где есть predict и target)
    # Если ты не сохранял OOF ансамбля, возьми OOF от LGBM или CatBoost для примера
    # Или сгенерируй OOF ансамбля, прогнав optimize_ensemble.py с сохранением
    try:
        # Для примера берем oof_catboost, но в идеале нужен oof ансамбля
        df = pd.read_csv(cfg.get_submission_path("oof_catboost.csv")) 
    except:
        print("❌ Нет файла OOF. Сначала обучи модели.")
        return

    y_true = df['target']
    y_pred = df['predict']

    # --- 1. MAPE (Средняя процентная ошибка) ---
    # Добавляем 1, чтобы не делить на 0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    print(f"\n🔹 MAPE (Средняя ошибка в %): {mape:.2f}%")

    # --- 2. Hit Rate (Попадание в диапазон) ---
    # Считаем, сколько предиктов попало в +/- 20% от факта
    threshold = 0.20
    error_ratio = np.abs(y_true - y_pred) / (y_true + 1)
    hit_rate = np.mean(error_ratio <= threshold) * 100
    print(f"🔹 Hit Rate @ 20% (Точность попадания): {hit_rate:.2f}% клиентов")

    # --- 3. Анализ по сегментам дохода ---
    print("\n🔹 Ошибка по сегментам (MAPE):")
    # Разбиваем на 3 группы: Низкий, Средний, Высокий доход
    df['segment'] = pd.qcut(df['target'], q=[0, 0.33, 0.66, 1.0], labels=['Low', 'Mid', 'High'])
    
    segment_stats = df.groupby('segment', observed=False).apply(
        lambda x: np.mean(np.abs((x['target'] - x['predict']) / (x['target'] + 1))) * 100
    )
    print(segment_stats)

    # --- 4. Риск-метрики (Недооценка vs Переоценка) ---
    overestimation = np.mean(y_pred > y_true) * 100
    underestimation = np.mean(y_pred <= y_true) * 100
    
    print(f"\n🔹 Баланс рисков:")
    print(f"   - Переоценка (Риск дефолта): {overestimation:.1f}% случаев")
    print(f"   - Недооценка (Упущенная выгода): {underestimation:.1f}% случаев")
    
    # --- 5. Корреляция Спирмена (Ранжирование) ---
    # Насколько хорошо модель отличает бедного от богатого
    spearman = df['target'].corr(df['predict'], method='spearman')
    print(f"\n🔹 Ранжирующая способность (Spearman Correlation): {spearman:.4f}")

if __name__ == "__main__":
    calculate_metrics()