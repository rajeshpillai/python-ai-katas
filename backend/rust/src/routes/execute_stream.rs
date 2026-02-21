use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use std::convert::Infallible;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;

use crate::config::Settings;
use crate::models::execution::ExecutionRequest;
use crate::services::preamble::SANDBOX_PREAMBLE;

const METRIC_PREFIX: &str = "__KATA_METRIC__:";
const TENSOR_PREFIX: &str = "__KATA_TENSOR__:";

pub async fn execute_stream(
    Json(req): Json<ExecutionRequest>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let settings = Settings::from_env();
    let timeout_secs = settings.sandbox_timeout_seconds;
    let code = req.code;

    let stream = async_stream::stream! {
        let start = Instant::now();
        let full_code = format!("{}\n{}", SANDBOX_PREAMBLE, code);

        // Create temp dir
        let tmp_dir = match tempfile::TempDir::new() {
            Ok(d) => d,
            Err(e) => {
                yield Ok(Event::default().event("error").data(format!("Failed to create temp directory: {}", e)));
                yield Ok(Event::default().event("done").data("0"));
                return;
            }
        };

        let source_path = tmp_dir.path().join("kata.rs");
        let binary_path = tmp_dir.path().join("kata");

        if let Err(e) = tokio::fs::write(&source_path, &full_code).await {
            yield Ok(Event::default().event("error").data(format!("Failed to write source: {}", e)));
            yield Ok(Event::default().event("done").data("0"));
            return;
        }

        // Compile
        let compile = Command::new("rustc")
            .args([
                source_path.to_str().unwrap(),
                "-o",
                binary_path.to_str().unwrap(),
                "--edition",
                "2021",
            ])
            .output()
            .await;

        match compile {
            Ok(output) if !output.status.success() => {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let cleaned = stderr.replace(source_path.to_str().unwrap_or(""), "<source>");
                yield Ok(Event::default().event("stderr").data(cleaned.clone()));
                yield Ok(Event::default().event("error").data(cleaned));
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                yield Ok(Event::default().event("done").data(format!("{:.2}", ms)));
                return;
            }
            Err(e) => {
                yield Ok(Event::default().event("error").data(format!("Failed to invoke rustc: {}", e)));
                yield Ok(Event::default().event("done").data("0"));
                return;
            }
            _ => {}
        }

        // Run the binary and stream output
        let mut child = match Command::new(&binary_path)
            .current_dir(tmp_dir.path())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                yield Ok(Event::default().event("error").data(format!("Failed to run binary: {}", e)));
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                yield Ok(Event::default().event("done").data(format!("{:.2}", ms)));
                return;
            }
        };

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let stdout_reader = BufReader::new(stdout);
        let stderr_reader = BufReader::new(stderr);

        let mut stdout_lines = LinesStream::new(stdout_reader.lines());
        let mut stderr_lines = LinesStream::new(stderr_reader.lines());

        // Stream stdout lines
        while let Some(line) = stdout_lines.next().await {
            let line: String = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if start.elapsed().as_secs() >= timeout_secs {
                let _ = child.kill().await;
                yield Ok(Event::default().event("error").data(format!(
                    "Execution timed out after {} seconds.",
                    timeout_secs
                )));
                break;
            }

            if line.starts_with(METRIC_PREFIX) {
                yield Ok(Event::default().event("metric").data(line));
            } else if line.starts_with(TENSOR_PREFIX) {
                yield Ok(Event::default().event("tensor").data(line));
            } else {
                yield Ok(Event::default().event("stdout").data(line));
            }
        }

        // Stream remaining stderr
        while let Some(line) = stderr_lines.next().await {
            let line: String = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            yield Ok(Event::default().event("stderr").data(line));
        }

        let ms = start.elapsed().as_secs_f64() * 1000.0;
        yield Ok(Event::default().event("done").data(format!("{:.2}", ms)));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
