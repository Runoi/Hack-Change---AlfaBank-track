import requests
import json
import random
import sys

# Адрес API (если запускаешь локально без докера, может быть localhost:8000)
# Если в докере, то обычно проброшен порт 8000
BASE_URL = "http://localhost:8000"

def run_tests():
    print(f"🚀 Запуск тестирования API: {BASE_URL}\n")

    # --- ТЕСТ 1: Получение списка клиентов ---
    print(f"📡 1. Запрос списка ID: GET /ClientsList")
    try:
        response = requests.get(f"{BASE_URL}/ClientsList")
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка: Не удалось подключиться к серверу. Он запущен?")
        sys.exit(1)

    if response.status_code == 200:
        data = response.json()
        ids = data.get("ids", [])
        count = len(ids)
        print(f"✅ Успех! Получено {count} ID.")
        
        if count > 0:
            print(f"   Примеры ID: {ids[:5]} ...")
        else:
            print("⚠️ Список ID пуст! Проверь, загрузился ли CSV файл в Docker.")
            sys.exit(1)
    else:
        print(f"❌ Ошибка {response.status_code}: {response.text}")
        sys.exit(1)

    # --- ТЕСТ 2: Анализ конкретного клиента ---
    # Берем случайный ID из списка
    target_id = random.choice(ids[:100]) # Берем из первых 100 для надежности
    
    print(f"\n📡 2. Запрос анализа по ID {target_id}: GET /ClientAnalysis/{target_id}")
    response = requests.get(f"{BASE_URL}/ClientAnalysis/{target_id}")

    if response.status_code == 200:
        data = response.json()
        print("✅ Успех! Ответ сервера:")
        # Красивый вывод JSON
        print(json.dumps(data, indent=4, ensure_ascii=False))
        
        print("\n   🔍 Проверка полей:")
        print(f"   - Predicted Income: {data.get('predicted_income')}")
        print(f"   - Breakdown: {data.get('model_breakdown')}")
        print(f"   - Top Factor: {data['explainability'][0]['feature']} ({data['explainability'][0]['impact']})")
    else:
        print(f"❌ Ошибка {response.status_code}: {response.text}")

    # --- ТЕСТ 3: Несуществующий клиент (Проверка обработки ошибок) ---
    fake_id = 999999999
    print(f"\n📡 3. Запрос несуществующего ID {fake_id} (Ожидаем 404)")
    response = requests.get(f"{BASE_URL}/ClientAnalysis/{fake_id}")

    if response.status_code == 404:
        print(f"✅ Тест пройден! Сервер вернул: {response.json()['detail']}")
    else:
        print(f"⚠️ Ожидалось 404, получено {response.status_code}")

if __name__ == "__main__":
    run_tests()