use axum::Json;

use crate::config::Settings;
use crate::models::execution::{ExecutionRequest, ExecutionResult};
use crate::services::sandbox;

pub async fn execute_code(Json(req): Json<ExecutionRequest>) -> Json<ExecutionResult> {
    let settings = Settings::from_env();
    let result = sandbox::execute_code(&req.code, settings.sandbox_timeout_seconds);
    Json(result)
}
