import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from catboost import CatBoostRegressor
import os
import pickle
import joblib
import torch
import torch.nn as nn

# --- АРХИТЕКТУРА НЕЙРОСЕТИ (Нужна для загрузки весов) ---
class ResBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout)
        )
    def forward(self, x): return x + self.block(x)

class TabularResNet(nn.Module):
    def __init__(self, cat_dims, num_cont, d_model, num_blocks, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, min(50, (c + 1) // 2)) for c in cat_dims])
        self.emb_dropout = nn.Dropout(dropout)
        total_emb_dim = sum(e.embedding_dim for e in self.embeddings) # type: ignore
        self.cont_bn = nn.BatchNorm1d(num_cont)
        self.cont_proj = nn.Linear(num_cont, d_model // 2)
        self.input_proj = nn.Sequential(nn.Linear(total_emb_dim + d_model // 2, d_model), nn.BatchNorm1d(d_model), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(d_model, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x_cat, x_cont):
        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, 1)
        x_cont = self.cont_proj(self.cont_bn(x_cont))
        x = self.input_proj(torch.cat([self.emb_dropout(x_emb), x_cont], 1))
        return self.head(self.blocks(x))
# --------------------------------------------------------

class InferenceService:
    def __init__(self, models_dir="models"):
        print(f"[INFO] Загрузка моделей из {models_dir}...")
        self.device = torch.device('cpu') # В докере используем CPU
        
        # --- LightGBM ---
        try:
            with open(os.path.join(models_dir, "lgbm_artifacts.pkl"), 'rb') as f:
                self.lgbm_artifacts = pickle.load(f)
            self.lgbm = joblib.load(os.path.join(models_dir, "model_lgbm.pkl"))
            self.explainer = shap.TreeExplainer(self.lgbm)
            print("✅ LGBM loaded")
        except Exception as e:
            print(f"⚠️ LGBM not loaded: {e}")
            self.lgbm = None

        # --- CatBoost ---
        try:
            with open(os.path.join(models_dir, "catboost_artifacts.pkl"), 'rb') as f:
                self.cat_artifacts = pickle.load(f)
            self.cat = CatBoostRegressor()
            self.cat.load_model(os.path.join(models_dir, "model_catboost.cbm"))
            print("✅ CatBoost loaded")
        except Exception as e:
            print(f"⚠️ CatBoost not loaded: {e}")
            self.cat = None

        # --- Neural Network (NEW) ---
        try:
            # Загружаем артефакты (скалер, энкодеры, конфиг сети)
            nn_path = os.path.join(models_dir, "model_nn_artifacts.pth")
            self.nn_artifacts = torch.load(nn_path, map_location=self.device, weights_only=False)
            
            # Инициализируем архитектуру
            self.nn_model = TabularResNet(
                cat_dims=self.nn_artifacts['cat_dims'],
                num_cont=len(self.nn_artifacts['cont_cols']),
                d_model=512,  # Должно совпадать с обучением
                num_blocks=3, # Должно совпадать с обучением
                dropout=0.15
            )
            # Загружаем веса
            self.nn_model.load_state_dict(self.nn_artifacts['model_state'])
            self.nn_model.to(self.device)
            self.nn_model.eval()
            print("✅ Neural Network loaded")
        except Exception as e:
            print(f"⚠️ Neural Network not loaded: {e}")
            self.nn_model = None

    def preprocess_and_align(self, df_raw, artifacts):
        """
        Единая функция предобработки для Бустингов.
        """
        model_features = artifacts['features']
        cat_features = artifacts.get('cat_features', []) 

        df = pd.DataFrame(columns=model_features)
        df = pd.concat([df, df_raw], ignore_index=True)
        
        for col in model_features:
            if col in cat_features:
                df[col] = df[col].astype(str).fillna("Unknown")
            else:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                # Бустинги умеют работать с NaN, но для надежности заполним нулями, где очевидно
                df[col] = df[col].fillna(0)

        return df[model_features]

    def predict(self, input_data: dict):
        df_raw = pd.DataFrame([input_data])
        breakdown = {}
        
        # --- 1. LGBM Inference ---
        df_lgbm = None
        if self.lgbm and hasattr(self, 'lgbm_artifacts'):
            try:
                df_lgbm = self.preprocess_and_align(df_raw, self.lgbm_artifacts)
                for col in self.lgbm_artifacts['cat_features']:
                    df_lgbm[col] = df_lgbm[col].astype('category')
                p = self.lgbm.predict(df_lgbm)[0]
                breakdown['lgbm'] = float(np.expm1(p))
            except Exception as e:
                print(f"❌ LGBM Error: {e}")
        
        # --- 2. CatBoost Inference ---
        if self.cat and hasattr(self, 'cat_artifacts'):
            try:
                df_cat = self.preprocess_and_align(df_raw, self.cat_artifacts)
                p = self.cat.predict(df_cat)[0]
                breakdown['catboost'] = float(np.expm1(p))
            except Exception as e:
                print(f"❌ CatBoost Error: {e}")

        # --- 3. Neural Network Inference ---
        if self.nn_model and hasattr(self, 'nn_artifacts'):
            try:
                # Специальный препроцессинг для NN
                cont_cols = self.nn_artifacts['cont_cols']
                cat_cols = self.nn_artifacts['cat_cols']
                scaler = self.nn_artifacts['scaler']
                encoders = self.nn_artifacts['label_encoders']

                # Числа: Чистим -> Заполняем 0 -> Скейлим
                df_nn_cont = df_raw.copy()
                for col in cont_cols:
                    if col not in df_nn_cont.columns:
                        df_nn_cont[col] = 0.0
                    else:
                        if df_nn_cont[col].dtype == 'object':
                            df_nn_cont[col] = pd.to_numeric(df_nn_cont[col].astype(str).str.replace(',', '.'), errors='coerce')
                        df_nn_cont[col] = df_nn_cont[col].fillna(0)
                
                # Масштабирование чисел
                X_cont = scaler.transform(df_nn_cont[cont_cols].values)
                X_cont_tensor = torch.tensor(X_cont, dtype=torch.float32).to(self.device)

                # Категории: Label Encoding
                X_cat = []
                for col in cat_cols:
                    val = str(df_raw.get(col, "Unknown").iloc[0])
                    le = encoders[col]
                    # Безопасный transform (если новая категория -> 0/Unknown)
                    if val in le.classes_:
                        idx = le.transform([val])[0]
                    else:
                        # Пытаемся найти Unknown или берем первый класс
                        idx = 0 
                    X_cat.append(idx)
                
                X_cat_tensor = torch.tensor([X_cat], dtype=torch.long).to(self.device)

                # Predict
                with torch.no_grad():
                    p_nn_log = self.nn_model(X_cat_tensor, X_cont_tensor).cpu().numpy()[0][0]
                
                breakdown['nn'] = float(np.expm1(p_nn_log))

            except Exception as e:
                print(f"❌ NN Error: {e}")

        if not breakdown:
            return {"error": "No models produced a prediction."}
            
        # --- Ансамбль (Оптимизированные веса) ---
        preds = []
        weights = []
        
        # Веса из AutoML: LGBM 70%, Cat 20%, NN 10%
        if 'lgbm' in breakdown:
            preds.append(breakdown['lgbm'])
            weights.append(0.80)
        if 'catboost' in breakdown:
            preds.append(breakdown['catboost'])
            weights.append(0.15)
        if 'nn' in breakdown:
            preds.append(breakdown['nn'])
            weights.append(0.05)
        
        # Взвешенное геометрическое среднее (лучше для доходов)
        # sum(w * log(p)) / sum(w) -> exp
        log_preds = np.log1p(preds)
        norm_weights = np.array(weights) / sum(weights)
        final_log = np.sum(log_preds * norm_weights)
        final_pred = float(np.expm1(final_log))
        
        # Клиппинг (Safety)
        final_pred = max(15000, min(final_pred, 3000000))
        
        # --- SHAP (LGBM only) ---
        explainability = []
        if df_lgbm is not None and hasattr(self, 'explainer'):
            try:
                shap_values = self.explainer.shap_values(df_lgbm)[0]
                for feat, val, impact in zip(df_lgbm.columns, df_lgbm.iloc[0], shap_values):
                    if abs(impact) > 0.05: 
                        explainability.append({"feature": str(feat), "value": str(val), "impact": float(impact)})
                explainability.sort(key=lambda x: abs(x['impact']), reverse=True)
            except Exception as e:
                print(f"❌ SHAP Error: {e}")

        return { "prediction": final_pred, "breakdown": breakdown, "shap": explainability[:7] }

service = InferenceService()