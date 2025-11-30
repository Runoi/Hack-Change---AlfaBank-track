import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  LabelList,
} from "recharts";
import type { WaterfallItem } from "../shared/types/types";

interface Props {
  data: WaterfallItem[];
}

const COLORS = ["#ef4444", "#b91c1c", "#fb7185", "#fca5a5"];

export const WaterfallChart: React.FC<Props> = ({ data }) => {
  return (
    <div style={{ minHeight: 300, height: 300, width: "100%" }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="horizontal"
          margin={{ top: 20, right: 24, left: 24, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="category" dataKey="name" />
          <YAxis type="number" />
          <Tooltip
            formatter={(value, name: string) => [
              Number(value).toFixed(2),
              name,
            ]}
            labelFormatter={(label) => `Item: ${label}`}
          />
          <Bar
            dataKey="offset"
            stackId="a"
            fill="transparent"
            isAnimationActive={false}
          />
          <Bar dataKey="delta" stackId="a" isAnimationActive={false}>
            {data.map((entry, idx) => (
              <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
            ))}
            <LabelList
              dataKey="delta"
              position="top"
              formatter={(value) => Number(value).toFixed(2)}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
