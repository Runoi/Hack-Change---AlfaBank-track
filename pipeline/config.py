from pydantic import BaseModel
from typing import List
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Paths(BaseModel):

    raw_data: str = str(BASE_DIR / "data")
    processed_data: str = str(BASE_DIR / "processed")
    
    submissions: str = str(BASE_DIR / "submissions")

class Files(BaseModel):
    train_raw: str = "hackathon_income_train.csv"
    test_raw: str = "hackathon_income_test.csv"
    train_proc: str = "train_processed.parquet"
    test_proc: str = "test_processed.parquet"

class Columns(BaseModel):
    id: str = "id"
    target: str = "target"
    weight: str = "w"
    drop: List[str] = [
        "id", "target", "w", "incomeValue", "incomeValueCategory", "dt",  "month"
    ]

# --- ПАРАМЕТРЫ МОДЕЛЕЙ ---

class CatBoostParams(BaseModel):
    iterations: int = 8000
    depth: int = 6       
    learning_rate: float = 0.03
    l2_leaf_reg: float = 3.0
    early_stopping: int = 500
    random_strength: float = 1.0 
    bagging_temperature: float = 1.0
    task_type: str = "GPU"
    loss_function: str = "MAE"

class LightGBMParams(BaseModel):
    num_leaves: int = 80            
    learning_rate: float = 0.015    
    num_boost_round: int = 10000    
    stopping_rounds: int = 3000      
    min_child_samples: int = 50     
    colsample_bytree: float = 0.7   
    subsample: float = 0.8         
    reg_alpha: float = 0.1          
    reg_lambda: float = 0.5         
    device: str = "gpu"

class NNParams(BaseModel):  
    epochs: int = 40
    batch_size: int = 1024
    max_lr: float = 0.004
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 512
    num_blocks: int = 3  

class EnsembleWeights(BaseModel):
    lgbm: float = 0.55
    catboost: float = 0.40
    nn: float = 0.05


class PipelineConfig(BaseModel):
    seed: int = 42
    folds: int = 5
    
    paths: Paths = Paths()
    files: Files = Files()
    cols: Columns = Columns()
    
    cat_params: CatBoostParams = CatBoostParams()
    lgbm_params: LightGBMParams = LightGBMParams()
    nn_params: NNParams = NNParams()
    
    weights: EnsembleWeights = EnsembleWeights()

    def get_train_raw_path(self) -> str:
        return os.path.join(self.paths.raw_data, self.files.train_raw)

    def get_test_raw_path(self) -> str:
        return os.path.join(self.paths.raw_data, self.files.test_raw)

    def get_train_proc_path(self) -> str:
        return os.path.join(self.paths.processed_data, self.files.train_proc)

    def get_test_proc_path(self) -> str:
        return os.path.join(self.paths.processed_data, self.files.test_proc)

    def get_submission_path(self, filename: str) -> str:
        return os.path.join(self.paths.submissions, filename)


cfg = PipelineConfig()