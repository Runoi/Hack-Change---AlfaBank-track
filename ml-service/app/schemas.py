from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ClientFeatures(BaseModel):
    client_id: Optional[str] = "unknown"
    features: Dict[str, Any]

class FeatureImportance(BaseModel):
    feature: str
    value: str
    impact: float  

# Модель оффера для API
class OfferResponse(BaseModel):
    product_code: str
    title: str
    client_message: str
    internal_comment: str
    priority: int

class PredictionResponse(BaseModel):
    client_id: str
    predicted_income: float
    model_breakdown: Dict[str, float]
    explainability: List[FeatureImportance]
    offers: List[OfferResponse] 

class ClientListResponse(BaseModel):
    ids: List[int]