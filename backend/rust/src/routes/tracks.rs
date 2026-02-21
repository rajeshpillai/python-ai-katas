use axum::Json;
use serde_json::{json, Value};

pub async fn list_tracks() -> Json<Value> {
    Json(json!([
        {
            "id": "foundational-ai",
            "name": "Foundational AI",
            "description": "Build deep intuition for AI from first principles — from data basics through neural networks to LLMs and reasoning models.",
            "status": "active"
        },
        {
            "id": "traditional-ai-ml",
            "name": "Traditional AI/ML",
            "description": "Classical AI, supervised and unsupervised learning, reinforcement learning, and traditional ML algorithms.",
            "status": "active"
        }
    ]))
}
