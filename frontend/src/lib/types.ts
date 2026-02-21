export interface PlotData {
  index: number;
  data: string;
  format: string;
}

export interface TensorData {
  name: string;
  shape: number[];
  values: number[][];
  min: number;
  max: number;
}

export interface ExecutionResult {
  stdout: string;
  stderr: string;
  error: string | null;
  execution_time_ms: number;
  metrics: Record<string, number | string>;
  plots: PlotData[];
  tensors: TensorData[];
}
