export interface SliderConfig {
  name: string;
  type: "float" | "int";
  min: number;
  max: number;
  defaultValue: number;
  step: number;
  lineIndex: number;
  assignmentLineIndex: number;
}

const PARAM_REGEX =
  /^#\s*@param\s+(\w+)\s+(float|int)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?$/;

export function parseSliderConfigs(code: string): SliderConfig[] {
  const lines = code.split("\n");
  const configs: SliderConfig[] = [];

  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].trim().match(PARAM_REGEX);
    if (match && i + 1 < lines.length) {
      const type = match[2] as "float" | "int";
      const step = match[6]
        ? parseFloat(match[6])
        : type === "int"
          ? 1
          : 0.001;
      configs.push({
        name: match[1],
        type,
        min: parseFloat(match[3]),
        max: parseFloat(match[4]),
        defaultValue: parseFloat(match[5]),
        step,
        lineIndex: i,
        assignmentLineIndex: i + 1,
      });
    }
  }
  return configs;
}

export function applySliderValues(
  code: string,
  configs: SliderConfig[],
  values: Record<string, number>,
): string {
  const lines = code.split("\n");
  for (const config of configs) {
    const val = values[config.name] ?? config.defaultValue;
    const formatted =
      config.type === "int" ? Math.round(val).toString() : val.toString();
    lines[config.assignmentLineIndex] = `${config.name} = ${formatted}`;
  }
  return lines.join("\n");
}
