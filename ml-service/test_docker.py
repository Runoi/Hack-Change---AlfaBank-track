import pandas as pd
import requests
import json
import os


TEST_CSV = "data/hackathon_income_test.csv"

def test_api():
    print("Читаем тестовый датасет...")
    if not os.path.exists(TEST_CSV):
        print(f"Файл {TEST_CSV} не найден!")
        return

    
    df = pd.read_csv(TEST_CSV, sep=';')
    sample_row = df.sample(1).iloc[0]
    
    
    features = sample_row.to_dict()
    
    features = {k: (str(v) if pd.isna(v) else v) for k, v in features.items()}

    payload = {
        "client_id": str(int(features.get('id', 0))),
        "features": features
    }
    
    print(f"\nОтправляем запрос для ID: {payload['client_id']}...")
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\nОтвет сервера:")
            print(json.dumps(data, indent=4, ensure_ascii=False))
            
            print(f"\n Предсказанный доход: {data['predicted_income']:,.2f} руб.")
            print("Топ-3 фактора влияния:")
            for item in data['explainability'][:3]:
                print(f"   - {item['feature']}: {item['impact']:+.2f}")
        else:
            print(f"\nОшибка сервера: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nНе удалось подключиться! Docker контейнер запущен?")

if __name__ == "__main__":
    test_api()