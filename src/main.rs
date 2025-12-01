//! StateSet Agents - Main Entry Point
//!
//! Starts all configured AI agents for autonomous commerce operations.

use stateset_agents::{AgentConfig, AgentManager};
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load environment variables
    let _ = dotenvy::dotenv();

    // Initialize logging
    let log_level = std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    let log_json = std::env::var("LOG_JSON").map(|v| v == "true").unwrap_or(false);

    if log_json {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| format!("stateset_agents={}", log_level).into()),
            )
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| format!("stateset_agents={}", log_level).into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    info!("Starting StateSet Agents v{}", stateset_agents::VERSION);

    // Load configuration
    let config = match AgentConfig::from_env() {
        Ok(config) => {
            info!("Configuration loaded successfully");
            info!("  API URL: {}", config.api.rest_url);
            info!("  gRPC URL: {:?}", config.api.grpc_url);
            info!("  Model: {}", config.openai.model);
            config
        }
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(e);
        }
    };

    // Create agent manager
    let mut manager = match AgentManager::new(config).await {
        Ok(manager) => {
            info!("Agent manager initialized");
            manager
        }
        Err(e) => {
            error!("Failed to initialize agent manager: {}", e);
            return Err(e);
        }
    };

    // Start all agents
    manager.start_all().await?;

    info!("All agents started. Press Ctrl+C to shutdown.");

    // Wait for shutdown signal
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, initiating shutdown...");
        }
        _ = async {
            #[cfg(unix)]
            {
                let mut sigterm = tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::terminate()
                ).expect("Failed to install SIGTERM handler");
                sigterm.recv().await;
            }
            #[cfg(not(unix))]
            {
                std::future::pending::<()>().await;
            }
        } => {
            info!("Received SIGTERM, initiating shutdown...");
        }
    }

    info!("StateSet Agents shutdown complete");
    Ok(())
}
