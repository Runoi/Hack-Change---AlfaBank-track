import { makeAutoObservable, runInAction } from "mobx";
import type { RawData } from "../shared/types/types";

class ClientStore {
  data: RawData | null = null;
  loading: boolean = false;
  error: string | null = null;

  constructor() {
    makeAutoObservable(this);
  }

  getClientAnalysis = async (id: string) => {
    runInAction(() => {
      this.loading = true;
      this.error = null;
      this.data = null;
    });

    try {
      const res = await fetch(`/api/ClientAnalysis/${encodeURIComponent(id)}`);

      if (!res.ok) {
        throw new Error(`Ошибка запроса: ${res.status}`);
      }

      const data = await res.json();
      runInAction(() => {
        this.data = data;
      });
    } catch (err) {
      runInAction(() => {
        this.error = "Неизвестная ошибка";
      });
      console.error(err);
    } finally {
      runInAction(() => {
        this.loading = false;
      });
    }
  };
}

export default ClientStore;
