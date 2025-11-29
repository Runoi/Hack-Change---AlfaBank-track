import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import os
import sys
import gc
import copy
import random
import warnings
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_float_col(series):
    if series.dtype == 'object' or series.dtype.name == 'category':
        return series.astype(str).str.replace(',', '.', regex=False).astype(float)
    return series

class TabularDataset(Dataset):
    def __init__(self, X_cat, X_cont, y=None, w=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        self.w = torch.tensor(w, dtype=torch.float32).unsqueeze(1) if w is not None else None
    def __len__(self): return len(self.X_cat)
    def __getitem__(self, idx):
        if self.y is not None: return self.X_cat[idx], self.X_cont[idx], self.y[idx], self.w[idx] # type: ignore
        return self.X_cat[idx], self.X_cont[idx]

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
        for emb in self.embeddings: nn.init.xavier_uniform_(emb.weight) # type: ignore
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

def train_model():
    seed_everything(cfg.seed)
    print(" Initializing ResNet Training Pipeline...")
    
    train_df = pd.read_parquet(cfg.get_train_proc_path())
    test_df = pd.read_parquet(cfg.get_test_proc_path())
    train_df[cfg.cols.target] = clean_float_col(train_df[cfg.cols.target])
    train_df[cfg.cols.weight] = clean_float_col(train_df[cfg.cols.weight])

    drop_cols = cfg.cols.drop.copy()
    if 'incomeValue' in test_df.columns:
        if test_df['incomeValue'].count() > 0:
            print(" Внимание: 'incomeValue' найден в тесте! Используем как фичу.")
            train_df['incomeValue'] = clean_float_col(train_df['incomeValue'])
            test_df['incomeValue'] = clean_float_col(test_df['incomeValue'])
            if 'incomeValue' in drop_cols: drop_cols.remove('incomeValue')

    features = [c for c in train_df.columns if c not in drop_cols and c in test_df.columns]
    cat_cols = [c for c in features if train_df[c].dtype.name == 'category' or train_df[c].dtype == 'object']
    cont_cols = [c for c in features if c not in cat_cols]
    
    print(f" Features: {len(cont_cols)} numeric, {len(cat_cols)} categorical")
    print(" NN: Заполнение пропусков нулями (локально)...")
    train_df[cont_cols] = train_df[cont_cols].fillna(0)
    test_df[cont_cols] = test_df[cont_cols].fillna(0)

    print(" Preprocessing: QuantileTransform...")
    scaler = QuantileTransformer(output_distribution='normal', random_state=42, subsample=200000)
    all_cont = np.concatenate([train_df[cont_cols].values, test_df[cont_cols].values], axis=0)
    scaler.fit(np.nan_to_num(all_cont, nan=0.0))
    
    X_cont = scaler.transform(np.nan_to_num(train_df[cont_cols].values))
    X_test_cont = scaler.transform(np.nan_to_num(test_df[cont_cols].values))
    
    X_cat = np.zeros((len(train_df), len(cat_cols)), dtype=int)
    X_test_cat = np.zeros((len(test_df), len(cat_cols)), dtype=int)
    
    cat_dims = []
    label_encoders = {}
    for i, col in enumerate(cat_cols):
        train_vals = train_df[col].astype(str).replace('nan', 'UNK').fillna("UNK").values
        test_vals = test_df[col].astype(str).replace('nan', 'UNK').fillna("UNK").values
        le = LabelEncoder()
        le.fit(np.unique(np.concatenate([train_vals, test_vals]))) # type: ignore
        X_cat[:, i] = le.transform(train_vals) # type: ignore
        X_test_cat[:, i] = le.transform(test_vals) # type: ignore
        cat_dims.append(len(le.classes_))
        label_encoders[col] = le 

    y_log = np.log1p(train_df[cfg.cols.target].values) # type: ignore
    w = train_df[cfg.cols.weight].values
    
    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    test_preds = np.zeros(len(test_df))
    oof_preds = np.zeros(len(train_df))
    scores = []
    
    print(f"\n Start Training on {DEVICE}")
    print(f"   Arch: ResNet ({cfg.nn_params.num_blocks} blocks), D_model: {cfg.nn_params.d_model}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cont, y_log)):
        train_ds = TabularDataset(X_cat[train_idx], X_cont[train_idx], y_log[train_idx], w[train_idx])
        val_ds = TabularDataset(X_cat[val_idx], X_cont[val_idx], y_log[val_idx], w[val_idx])
        train_loader = DataLoader(train_ds, batch_size=cfg.nn_params.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.nn_params.batch_size*2, num_workers=0, pin_memory=True)
        
        model = TabularResNet(cat_dims, len(cont_cols), cfg.nn_params.d_model, cfg.nn_params.num_blocks, cfg.nn_params.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.nn_params.max_lr, weight_decay=cfg.nn_params.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.nn_params.max_lr, steps_per_epoch=len(train_loader), epochs=cfg.nn_params.epochs, pct_start=0.2, div_factor=25.0, final_div_factor=1000.0)
        criterion = nn.L1Loss(reduction='none')
        
        best_wmae = float('inf')
        best_weights = None
        loop = tqdm(range(cfg.nn_params.epochs), desc=f"Fold {fold+1}", leave=False)
        
        for epoch in loop:
            model.train()
            train_loss_accum = 0
            for x_cat_b, x_cont_b, y_b, w_b in train_loader:
                x_cat_b, x_cont_b, y_b, w_b = x_cat_b.to(DEVICE), x_cont_b.to(DEVICE), y_b.to(DEVICE), w_b.to(DEVICE)
                optimizer.zero_grad()
                preds = model(x_cat_b, x_cont_b)
                loss = (criterion(preds, y_b) * w_b).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss_accum += loss.item()
            
            model.eval()
            val_preds, val_targets, val_weights = [], [], []
            with torch.no_grad():
                for x_cat_b, x_cont_b, y_b, w_b in val_loader:
                    x_cat_b, x_cont_b = x_cat_b.to(DEVICE), x_cont_b.to(DEVICE)
                    val_preds.append(model(x_cat_b, x_cont_b).cpu().numpy())
                    val_targets.append(y_b.numpy())
                    val_weights.append(w_b.numpy())
            
            val_p = np.clip(np.concatenate(val_preds).flatten(), 0, 18)
            val_t = np.concatenate(val_targets).flatten()
            val_w = np.concatenate(val_weights).flatten()
            wmae = np.average(np.abs(np.expm1(val_p) - np.expm1(val_t)), weights=val_w)
            
            if wmae < best_wmae:
                best_wmae = wmae
                best_weights = copy.deepcopy(model.state_dict())
            loop.set_postfix(val_wmae=f"{wmae:,.0f}")
        
        scores.append(best_wmae)
        print(f"   Fold {fold+1} Best WMAE: {best_wmae:,.2f}")
        
        if best_weights: model.load_state_dict(best_weights)
        
        model.eval()
        oof_preds_fold = []
        with torch.no_grad():
            for x_cat_b, x_cont_b, _, _ in val_loader:
                x_cat_b, x_cont_b = x_cat_b.to(DEVICE), x_cont_b.to(DEVICE)
                oof_preds_fold.append(model(x_cat_b, x_cont_b).cpu().numpy())
        oof_preds[val_idx] = np.expm1(np.clip(np.concatenate(oof_preds_fold).flatten(), 0, 18))

        test_ds = TabularDataset(X_test_cat, X_test_cont)
        test_loader = DataLoader(test_ds, batch_size=cfg.nn_params.batch_size*2)
        f_preds = []
        with torch.no_grad():
            for x_cat_b, x_cont_b in test_loader:
                x_cat_b, x_cont_b = x_cat_b.to(DEVICE), x_cont_b.to(DEVICE)
                f_preds.append(model(x_cat_b, x_cont_b).cpu().numpy())
        test_preds += np.clip(np.concatenate(f_preds).flatten(), 0, 18) / cfg.folds
        
        if fold == cfg.folds - 1:
            artifacts = {
                'model_state': model.state_dict(),
                'cat_dims': cat_dims,
                'cont_cols': cont_cols,
                'cat_cols': cat_cols,
                'scaler': scaler,
                'label_encoders': label_encoders 
            }
            torch.save(artifacts, cfg.get_submission_path("model_nn_artifacts.pth"))
            print(" Артефакты NN для Docker сохранены.")

        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    final_test_preds = np.expm1(test_preds)
    
    sub_path = cfg.get_submission_path("submission_nn.csv")
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    pd.DataFrame({cfg.cols.id: test_df[cfg.cols.id], 'predict': final_test_preds}).to_csv(sub_path, index=False)
    
    oof_path = cfg.get_submission_path("oof_nn.csv")
    pd.DataFrame({
        cfg.cols.id: train_df[cfg.cols.id],
        'target': train_df[cfg.cols.target],
        'w': train_df[cfg.cols.weight],
        'predict': oof_preds
    }).to_csv(oof_path, index=False)
    
    print(f"\n ResNet Avg WMAE: {np.mean(scores):,.2f}")
    print(f" Сабмит: {sub_path}")

if __name__ == "__main__":
    train_model()