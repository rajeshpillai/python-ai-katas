mod config;
mod models;
mod routes;
mod services;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use config::Settings;

#[tokio::main]
async fn main() {
    let settings = Settings::from_env();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api_routes = Router::new()
        .route("/tracks", get(routes::tracks::list_tracks))
        .route(
            "/tracks/{track_id}/katas",
            get(routes::katas::list_katas),
        )
        .route(
            "/tracks/{track_id}/katas/{phase_id}/{kata_id}/content",
            get(routes::katas::get_kata_content),
        )
        .route("/execute", post(routes::execute::execute_code))
        .route(
            "/execute/stream",
            post(routes::execute_stream::execute_stream),
        );

    let app = Router::new()
        .route("/health", get(routes::health::health_check))
        .nest("/api", api_routes)
        .layer(cors);

    let addr = format!("0.0.0.0:{}", settings.port);
    println!("Rust AI Katas backend listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
