import { For } from "solid-js";
import type { SliderConfig } from "../../lib/slider-config";
import "./slider-bar.css";

interface SliderBarProps {
  configs: SliderConfig[];
  values: Record<string, number>;
  onChange: (name: string, value: number) => void;
}

export default function SliderBar(props: SliderBarProps) {
  return (
    <div class="slider-bar">
      <For each={props.configs}>
        {(config) => (
          <div class="slider-bar__item">
            <label class="slider-bar__label">
              {config.name.replace(/_/g, " ")}
            </label>
            <input
              type="range"
              class="slider-bar__input"
              min={config.min}
              max={config.max}
              step={config.step}
              value={props.values[config.name] ?? config.defaultValue}
              onInput={(e) =>
                props.onChange(config.name, parseFloat(e.currentTarget.value))
              }
            />
            <span class="slider-bar__value">
              {(props.values[config.name] ?? config.defaultValue).toFixed(
                config.type === "int" ? 0 : 3,
              )}
            </span>
          </div>
        )}
      </For>
    </div>
  );
}
