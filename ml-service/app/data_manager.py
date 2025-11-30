import pandas as pd
import os
import sys

DATA_PATH = os.path.join("data", "hackathon_income_test.csv")

class DataManager:
    def __init__(self):
        self.df = None
        self.load_data()

    def load_data(self):
        print(f"[INFO] Загрузка данных клиентов из {DATA_PATH}...")
        if os.path.exists(DATA_PATH):
            try:
                self.df = pd.read_csv(DATA_PATH, sep=';')
                
                if 'id' in self.df.columns:
                    self.df['id'] = self.df['id'].astype(int)
                    self.df.set_index('id', inplace=False)
                
                print(f" Данные загружены. Клиентов: {len(self.df)}")
            except Exception as e:
                print(f" Ошибка чтения CSV: {e}")
        else:
            print(f" Файл {DATA_PATH} не найден! Эндпоинты клиентов не будут работать.")

    def get_all_ids(self) -> list:
        if self.df is not None and 'id' in self.df.columns:
            return self.df['id'].tolist()
        return []

    def get_client_features(self, client_id: int) -> dict:
        if self.df is None:
            return None # type: ignore
        
        client_row = self.df[self.df['id'] == client_id]
        
        if client_row.empty:
            return None # type: ignore
        
        record = client_row.iloc[0].to_dict()
        
        clean_record = {k: (v if pd.notna(v) else None) for k, v in record.items()}
        
        return clean_record

data_manager = DataManager()