use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ExecutionRequest {
    pub code: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub execution_time_ms: f64,
    pub metrics: HashMap<String, serde_json::Value>,
    pub plots: Vec<serde_json::Value>,
    pub tensors: Vec<serde_json::Value>,
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            error: None,
            execution_time_ms: 0.0,
            metrics: HashMap::new(),
            plots: Vec::new(),
            tensors: Vec::new(),
        }
    }
}
