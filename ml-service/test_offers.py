import requests
import json

# Адрес API (убедись, что Docker запущен)
URL = "http://localhost:8000/predict"

def test_scenario(name, features):
    print(f"\n🧪 ТЕСТ: {name}")
    print("-" * 60)
    
    # Эмулируем запрос
    payload = {
        "client_id": "test_user",
        "features": features
    }
    
    try:
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            income = data['predicted_income']
            offers = data['offers']
            
            print(f"💰 Прогноз дохода: {income:,.0f} руб.")
            
            if offers:
                print(f"🎉 ПРЕДЛОЖЕНИЯ ({len(offers)} шт):")
                for i, offer in enumerate(offers, 1):
                    print(f"\n   #{i} [{offer['product_code']}] Приоритет: {offer['priority']}")
                    print(f"   📢 Клиенту:  {offer['client_message']}")
                    print(f"   🕵️  CRM-info: {offer['internal_comment']}")
            else:
                print("❌ Нет предложений (условия не сработали).")
        else:
            print(f"Ошибка API: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("⛔ Не удалось подключиться. Проверь, запущен ли Docker!")

def main():
    # --- СЦЕНАРИЙ 1: Идеальный кандидат на Кредитку ---
    # Доход есть (эмулируем через salary_6to12m_avg), кредитки нет, долгов нет.
    features_cc = {
        "salary_6to12m_avg": 85000,          # Прогноз будет около этой суммы
        "hdb_bki_active_cc_max_limit": 0,    # Нет кредитки
        "hdb_outstand_sum": 0,               # Нет долгов
        "age": 28
    }
    test_scenario("Кандидат на Кредитную Карту", features_cc)

    # --- СЦЕНАРИЙ 2: Ипотечный клиент ---
    # Высокий доход, возраст ОК, ипотеки нет.
    features_mortgage = {
        "salary_6to12m_avg": 160000,
        "hdb_bki_total_ip_cnt": 0,           # Нет ипотеки
        "age": 32
    }
    test_scenario("Молодая семья (Ипотека)", features_mortgage)

    # --- СЦЕНАРИЙ 3: VIP Клиент ---
    # Очень высокий доход.
    features_vip = {
        "salary_6to12m_avg": 500000,
        "age": 45
    }
    test_scenario("VIP Клиент (Alfa Premium)", features_vip)

    # --- СЦЕНАРИЙ 4: Путешественник ---
    # Доход средний, но большие траты на туризм.
    features_travel = {
        "salary_6to12m_avg": 90000,
        "avg_6m_travel": 25000,              # Триггер Travel
        "hdb_bki_active_cc_max_limit": 10000 # Кредитка уже есть (чтобы не сработал Rule 1)
    }
    test_scenario("Путешественник (Travel Card)", features_travel)

    # --- СЦЕНАРИЙ 5: Забота о здоровье (Налоговый вычет) ---
    # Тратит на врачей и платит налоги.
    features_health = {
        "salary_6to12m_avg": 70000,
        "avg_3m_healthcare_services": 15000, # Много трат на медицину
        "avg_6m_government_services": 2000,  # Платит налоги
        "hdb_bki_active_cc_max_limit": 50000 # Кредитка есть
    }
    test_scenario("Пациент (Налоговый вычет)", features_health)

if __name__ == "__main__":
    main()