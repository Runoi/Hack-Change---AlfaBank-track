import { createContext } from "react";
import ClientStore from "./ClientStore";

export class RootStore {
  ClientStore = new ClientStore();
}

export const rootStore = new RootStore();

export const RootStoreContext = createContext(rootStore);
