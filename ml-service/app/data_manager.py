import pandas as pd
import os
import sys

# Путь к данным внутри контейнера или локально
# Учитываем, что main.py запускается из корня ml-service
DATA_PATH = os.path.join("data", "hackathon_income_test.csv")

class DataManager:
    def __init__(self):
        self.df = None
        self.load_data()

    def load_data(self):
        print(f"[INFO] Загрузка данных клиентов из {DATA_PATH}...")
        if os.path.exists(DATA_PATH):
            try:
                # Читаем CSV (разделитель точка с запятой, как в твоем примере)
                self.df = pd.read_csv(DATA_PATH, sep=';')
                
                # Убедимся, что ID - это строка или int, для поиска
                # Если в CSV есть колонка 'id'
                if 'id' in self.df.columns:
                    self.df['id'] = self.df['id'].astype(int)
                    # Индексируем по ID для быстрого поиска
                    self.df.set_index('id', inplace=False) 
                
                print(f"✅ Данные загружены. Клиентов: {len(self.df)}")
            except Exception as e:
                print(f"❌ Ошибка чтения CSV: {e}")
        else:
            print(f"⚠️ Файл {DATA_PATH} не найден! Эндпоинты клиентов не будут работать.")

    def get_all_ids(self) -> list:
        if self.df is not None and 'id' in self.df.columns:
            # Возвращаем список ID (первые 1000 или все, чтобы не перегрузить фронт)
            # Для продакшена лучше пагинация, но для хакатона вернем все или срез
            return self.df['id'].tolist()
        return []

    def get_client_features(self, client_id: int) -> dict:
        if self.df is None:
            return None # type: ignore
        
        # Ищем клиента
        client_row = self.df[self.df['id'] == client_id]
        
        if client_row.empty:
            return None # type: ignore
        
        # Превращаем в словарь
        record = client_row.iloc[0].to_dict()
        
        # Важно: FastAPI не любит NaN (float("nan")), заменяем на None
        clean_record = {k: (v if pd.notna(v) else None) for k, v in record.items()}
        
        return clean_record

# Создаем экземпляр
data_manager = DataManager()