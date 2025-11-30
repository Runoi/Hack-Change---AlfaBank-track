import type { RadarPoint, RawData, WaterfallItem } from "../types/types";

export function prepareWaterfall(data: RawData): WaterfallItem[] {
  const items = Object.entries(data.model_breakdown).map(([name, value]) => ({
    name,
    value,
  }));
  let cumulative = 0;
  const names = ["LightBoost", "CatBoost", "NeuralNetwork"];
  return items.map((it, index) => {
    const offset = cumulative;
    cumulative += it.value;
    return {
      name: names[index],
      offset,
      delta: it.value,
      totalAfter: cumulative,
    } as WaterfallItem;
  });
}

export function prepareRadar(data: RawData): RadarPoint[] {
  return data.explainability.map((e) => ({
    feature: e.feature,
    impact: Number(e.impact),
  }));
}
