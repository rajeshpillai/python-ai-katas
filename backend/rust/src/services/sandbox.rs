use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;
use tempfile::TempDir;

use crate::models::execution::ExecutionResult;
use crate::services::preamble::SANDBOX_PREAMBLE;

const METRIC_PREFIX: &str = "__KATA_METRIC__:";
const METRIC_SUFFIX: &str = ":__END_KATA_METRIC__";
const TENSOR_PREFIX: &str = "__KATA_TENSOR__:";
const TENSOR_SUFFIX: &str = ":__END_KATA_TENSOR__";
pub fn execute_code(code: &str, timeout_seconds: u64) -> ExecutionResult {
    let start = Instant::now();

    let full_code = format!("{}\n{}", SANDBOX_PREAMBLE, code);

    // Create temp directory for compilation
    let tmp_dir = match TempDir::new() {
        Ok(d) => d,
        Err(e) => {
            return ExecutionResult {
                error: Some(format!("Failed to create temp directory: {}", e)),
                execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                ..Default::default()
            };
        }
    };

    let source_path = tmp_dir.path().join("kata.rs");
    let binary_path = tmp_dir.path().join("kata");

    if let Err(e) = std::fs::write(&source_path, &full_code) {
        return ExecutionResult {
            error: Some(format!("Failed to write source file: {}", e)),
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        };
    }

    // Compile
    let compile_result = Command::new("rustc")
        .args([
            source_path.to_str().unwrap(),
            "-o",
            binary_path.to_str().unwrap(),
            "--edition",
            "2021",
        ])
        .output();

    let compile_output = match compile_result {
        Ok(o) => o,
        Err(e) => {
            return ExecutionResult {
                error: Some(format!("Failed to invoke rustc: {}. Is Rust installed?", e)),
                execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                ..Default::default()
            };
        }
    };

    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr).to_string();
        // Clean up temp paths from error messages for readability
        let cleaned = stderr.replace(source_path.to_str().unwrap_or(""), "<source>");
        return ExecutionResult {
            stderr: cleaned.clone(),
            error: Some(cleaned),
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        };
    }

    // Run the compiled binary with timeout
    let run_start = Instant::now();
    let run_result = Command::new(&binary_path)
        .current_dir(tmp_dir.path())
        .output();

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let run_output = match run_result {
        Ok(o) => o,
        Err(e) => {
            if run_start.elapsed().as_secs() >= timeout_seconds {
                return ExecutionResult {
                    error: Some(format!(
                        "Execution timed out after {} seconds. Your code took too long to run. Try reducing the number of iterations or data size.",
                        timeout_seconds
                    )),
                    execution_time_ms: elapsed_ms,
                    ..Default::default()
                };
            }
            return ExecutionResult {
                error: Some(format!("Failed to run binary: {}", e)),
                execution_time_ms: elapsed_ms,
                ..Default::default()
            };
        }
    };

    let raw_stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&run_output.stderr).to_string();

    // Parse sentinel markers from stdout
    let mut clean_stdout = String::new();
    let mut metrics: HashMap<String, serde_json::Value> = HashMap::new();
    let mut tensors: Vec<serde_json::Value> = Vec::new();

    for line in raw_stdout.lines() {
        if let Some(rest) = line.strip_prefix(METRIC_PREFIX) {
            if let Some(payload) = rest.strip_suffix(METRIC_SUFFIX) {
                if let Some((name, val_str)) = payload.split_once(':') {
                    let val: serde_json::Value = val_str
                        .parse::<f64>()
                        .map(|v| serde_json::json!(v))
                        .unwrap_or_else(|_| serde_json::json!(val_str));
                    metrics.insert(name.to_string(), val);
                }
            }
        } else if let Some(rest) = line.strip_prefix(TENSOR_PREFIX) {
            if let Some(payload) = rest.strip_suffix(TENSOR_SUFFIX) {
                if let Ok(tensor) = serde_json::from_str::<serde_json::Value>(payload) {
                    tensors.push(tensor);
                }
            }
        } else {
            if !clean_stdout.is_empty() {
                clean_stdout.push('\n');
            }
            clean_stdout.push_str(line);
        }
    }

    let error = if !run_output.status.success() {
        Some(if stderr.is_empty() {
            format!("Process exited with code: {:?}", run_output.status.code())
        } else {
            stderr.clone()
        })
    } else {
        None
    };

    ExecutionResult {
        stdout: clean_stdout,
        stderr,
        error,
        execution_time_ms: elapsed_ms,
        metrics,
        plots: Vec::new(),
        tensors,
    }
}
