use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::models::kata::{get_track, KataListResponse};

fn content_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    // Try relative to CWD first (development), then relative to exe
    let cwd_content = std::env::current_dir()
        .map(|d| d.join("content"))
        .unwrap_or_default();

    if cwd_content.exists() {
        return cwd_content;
    }

    if let Some(dir) = exe_dir {
        let exe_content = dir.join("content");
        if exe_content.exists() {
            return exe_content;
        }
    }

    cwd_content
}

pub async fn list_katas(Path(track_id): Path<String>) -> Response {
    let track = match get_track(&track_id) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"detail": format!("Track '{}' not found", track_id)})),
            )
                .into_response();
        }
    };

    let phases: HashMap<&str, &str> = track.phases.iter().copied().collect();

    let response = KataListResponse {
        katas: track.katas.iter().collect(),
        phases,
    };

    Json(response).into_response()
}

pub async fn get_kata_content(
    Path((track_id, phase_id, kata_id)): Path<(String, u32, String)>,
) -> Response {
    let track = match get_track(&track_id) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"detail": format!("Track '{}' not found", track_id)})),
            )
                .into_response();
        }
    };

    let kata = track
        .katas
        .iter()
        .find(|k| k.id == kata_id && k.phase == phase_id);

    let kata = match kata {
        Some(k) => k,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "detail": format!("Kata '{}' not found in phase {}", kata_id, phase_id)
                })),
            )
                .into_response();
        }
    };

    let filename = format!("{:02}-{}.md", kata.sequence, kata_id);
    let filepath = content_dir()
        .join(&track_id)
        .join(format!("phase-{}", phase_id))
        .join(&filename);

    match std::fs::read_to_string(&filepath) {
        Ok(content) => (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            content,
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "detail": format!("Content file not found: {}", filename)
            })),
        )
            .into_response(),
    }
}
