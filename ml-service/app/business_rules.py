from typing import List, Dict, Any

def generate_offers(features: Dict[str, Any], predicted_income: float) -> List[Dict[str, Any]]:
    """
    Генератор персональных предложений (Next Best Action).
    Гарантирует выдачу офферов для любого сегмента (от VIP до Mass).
    """
    offers = []
    
    def get_val(key, default=0.0):
        val = features.get(key)
        if val is None or str(val).lower() in ['nan', 'none', 'null', '']:
            return default
        try:
            return float(val)
        except:
            return default

    total_debt = get_val('hdb_outstand_sum') + get_val('hdb_other_outstand_sum')
    dti_ratio = total_debt / (predicted_income + 1.0) if predicted_income > 0 else 999
    
    has_cc = get_val('hdb_bki_active_cc_max_limit') > 0
    has_mortgage = get_val('hdb_bki_total_ip_cnt') > 0
    
    travel_spend = get_val('avg_6m_travel') + get_val('avg_by_category__amount__sum__cashflowcategory_name__puteshestvija')
    health_spend = get_val('avg_3m_healthcare_services')
    gov_spend = get_val('avg_6m_government_services')
    
    age = get_val('age')
    if age < 18: age = 30

    if not has_cc and predicted_income > 40000 and dti_ratio < 15.0:
        limit = min(predicted_income * 3, 500000)
        offers.append({
            "product_code": "CC_100_DAYS",
            "title": "Кредитка: Год без %",
            "internal_comment": f"Cross-sell. Доход {predicted_income:.0f}, нагрузка приемлемая (DTI={dti_ratio:.1f}).",
            "client_message": f"Вам предварительно одобрен лимит {limit:,.0f} . Год без процентов на всё.",
            "priority": 90
        })

    if not has_mortgage and predicted_income > 100000 and 21 <= age <= 60:
        max_amt = predicted_income * 0.45 * 12 * 20
        offers.append({
            "product_code": "MORTGAGE_PRIMARY",
            "title": "Ипотека от 5.9%",
            "internal_comment": "Upsell. Высокий доход, нет текущей ипотеки.",
            "client_message": f"Рассчитали для вас лимит: до {max_amt/1000000:.1f} млн на покупку жилья.",
            "priority": 80
        })

    if predicted_income > 300000:
        offers.append({
            "product_code": "ALFA_PREMIUM",
            "title": "Alfa Premium",
            "internal_comment": "Retention VIP. Высокий прогноз дохода.",
            "client_message": "Премиальное обслуживание, бизнес-залы и трансферы. Попробуйте бесплатно.",
            "priority": 100
        })

    if travel_spend > 3000 or predicted_income > 120000:
        offers.append({
            "product_code": "ALFA_TRAVEL",
            "title": "Карта Alfa Travel",
            "internal_comment": "Ecosystem. Активный путешественник или High Mass сегмент.",
            "client_message": "До 10% милями за покупки и бесплатная страховка для визы.",
            "priority": 70
        })

    if health_spend > 2000 or gov_spend > 0:
        offers.append({
            "product_code": "TAX_HELPER",
            "title": "Верните налоги (НДФЛ)",
            "internal_comment": "Service. Есть траты на медицину или налоги.",
            "client_message": "Поможем оформить налоговый вычет за лечение, обучение или фитнес.",
            "priority": 60
        })

    if dti_ratio >= 15.0 and total_debt > 50000:
        offers.append({
            "product_code": "REFINANCE",
            "title": "Снижение платежа",
            "internal_comment": "Risk Retention. Высокая закредитованность. Предложить рефинансирование.",
            "client_message": "Объедините кредиты в один и платите меньше каждый месяц.",
            "priority": 95
        })

    if predicted_income <= 300000:
        offers.append({
            "product_code": "DEBIT_CASHBACK",
            "title": "Лучший кэшбэк",
            "internal_comment": "Base Product. Предлагаем всем в Mass сегменте.",
            "client_message": "Бесплатная карта. Кэшбэк на супермаркеты, кафе и такси каждый месяц.",
            "priority": 50
        })

    if predicted_income > 20000:
        offers.append({
            "product_code": "SAVE_ACCOUNT",
            "title": "Накопительный счет",
            "internal_comment": "Funding. Пассивное привлечение средств.",
            "client_message": "До 16% годовых на остаток с первого месяца. Снимайте когда удобно.",
            "priority": 40
        })

    offers.sort(key=lambda x: x["priority"], reverse=True)
    return offers[:4]