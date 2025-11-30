from fastapi import FastAPI, HTTPException
from typing import List
from .schemas import ClientFeatures, PredictionResponse, ClientListResponse
from .inference import service
from .data_manager import data_manager
from .business_rules import generate_offers

app = FastAPI(title="Alfa-Bank Income Prediction", version="1.0")

@app.post("/predict", response_model=PredictionResponse)
def predict_income(data: ClientFeatures):
    result = service.predict(data.features)
    
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    offers = generate_offers(data.features, result["prediction"]) # type: ignore

    return {
        "client_id": str(data.client_id),
        "predicted_income": result["prediction"],
        "model_breakdown": result["breakdown"],
        "explainability": result["shap"],
        "offers": offers
    }

@app.get("/ClientsList", response_model=ClientListResponse)
def get_clients_list():
    ids = data_manager.get_all_ids()
    return {"ids": ids[:5000]}

@app.get("/ClientAnalysis/{client_id}", response_model=PredictionResponse)
def analyze_client_by_id(client_id: int):
    features = data_manager.get_client_features(client_id)
    
    if not features:
        raise HTTPException(status_code=404, detail=f"Client ID {client_id} not found")
    
    result = service.predict(features)
    
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    offers = generate_offers(features, result["prediction"]) # type: ignore

    return {
        "client_id": str(client_id),
        "predicted_income": result["prediction"],
        "model_breakdown": result["breakdown"],
        "explainability": result["shap"],
        "offers": offers
    }

@app.get("/health")
def health():
    return {"status": "ok"}