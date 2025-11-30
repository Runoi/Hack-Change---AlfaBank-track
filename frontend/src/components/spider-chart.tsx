import React from "react";
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Tooltip,
  Radar,
} from "recharts";
import type { RadarPoint } from "../shared/types/types";

interface Props {
  data: RadarPoint[];
}

export const RadarExplain: React.FC<Props> = ({ data }) => {
  return (
    <div style={{ minHeight: 300, height: 300, width: "100%" }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="feature" tick={{ fontSize: 12 }} />
          <Tooltip formatter={(value) => Number(value).toFixed(3)} />
          <Radar
            name="impact"
            dataKey="impact"
            stroke="#c40000"
            fill="#c40000"
            fillOpacity={0.3}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};
