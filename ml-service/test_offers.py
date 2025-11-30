import requests
import json

URL = "http://localhost:8000/predict"

def test_scenario(name, features):
    print(f"\n ТЕСТ: {name}")
    print("-" * 60)
    
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
            
            print(f" Прогноз дохода: {income:,.0f} руб.")
            
            if offers:
                print(f" ПРЕДЛОЖЕНИЯ ({len(offers)} шт):")
                for i, offer in enumerate(offers, 1):
                    print(f"\n   #{i} [{offer['product_code']}] Приоритет: {offer['priority']}")
                    print(f"   Клиенту:  {offer['client_message']}")
                    print(f"   CRM-info: {offer['internal_comment']}")
            else:
                print(" Нет предложений (условия не сработали).")
        else:
            print(f"Ошибка API: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(" Не удалось подключиться. Проверь, запущен ли Docker!")

def main():
    features_cc = {
        "salary_6to12m_avg": 85000,
        "hdb_bki_active_cc_max_limit": 0,
        "hdb_outstand_sum": 0,
        "age": 28
    }
    test_scenario("Кандидат на Кредитную Карту", features_cc)

    features_mortgage = {
        "salary_6to12m_avg": 160000,
        "hdb_bki_total_ip_cnt": 0,
        "age": 32
    }
    test_scenario("Молодая семья (Ипотека)", features_mortgage)

    features_vip = {
        "salary_6to12m_avg": 500000,
        "age": 45
    }
    test_scenario("VIP Клиент (Alfa Premium)", features_vip)

    features_travel = {
        "salary_6to12m_avg": 90000,
        "avg_6m_travel": 25000,
        "hdb_bki_active_cc_max_limit": 10000
    }
    test_scenario("Путешественник (Travel Card)", features_travel)

    features_health = {
        "salary_6to12m_avg": 70000,
        "avg_3m_healthcare_services": 15000,
        "avg_6m_government_services": 2000,
        "hdb_bki_active_cc_max_limit": 50000
    }
    test_scenario("Пациент (Налоговый вычет)", features_health)

if __name__ == "__main__":
    main()