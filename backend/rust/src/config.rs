use std::env;

pub struct Settings {
    pub sandbox_timeout_seconds: u64,
    pub port: u16,
}

impl Settings {
    pub fn from_env() -> Self {
        let sandbox_timeout_seconds = env::var("KATAS_SANDBOX_TIMEOUT_SECONDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let port = env::var("KATAS_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8001);

        Self {
            sandbox_timeout_seconds,
            port,
        }
    }
}
