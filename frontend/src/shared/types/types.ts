export interface ModelBreakdown {
  [modelName: string]: number;
}

export interface ExplainabilityItem {
  feature: string;
  value: string | number;
  impact: number;
}

export interface RawData {
  client_id: string;
  predicted_income: number;
  model_breakdown: ModelBreakdown;
  explainability: ExplainabilityItem[];
  offers: {
    product_code: string;
    title: string;
    client_message: string;
    internal_comment: string;
    priority: number;
  }[];
}

export interface WaterfallItem {
  name: string;
  offset: number;
  delta: number;
  totalAfter: number;
}

export interface RadarPoint {
  feature: string;
  impact: number;
}
