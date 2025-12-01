//! Configuration module for StateSet Agents
//!
//! Supports loading from environment variables, config files, and programmatic setup.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main configuration for the StateSet Agents framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// StateSet API configuration
    pub api: ApiConfig,

    /// OpenAI configuration for LLM-powered agents
    pub openai: OpenAIConfig,

    /// Agent-specific configurations
    pub agents: AgentsConfig,

    /// Redis configuration (optional)
    #[serde(default)]
    pub redis: Option<RedisConfig>,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// Configuration for connecting to the StateSet API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Base URL for REST API (e.g., "http://localhost:8080/api/v1")
    pub rest_url: String,

    /// gRPC endpoint (e.g., "http://localhost:8081")
    #[serde(default)]
    pub grpc_url: Option<String>,

    /// API key for authentication
    pub api_key: String,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum retry attempts
    #[serde(default = "default_retries")]
    pub max_retries: u32,

    /// Whether to prefer gRPC over REST when available
    #[serde(default)]
    pub prefer_grpc: bool,
}

/// OpenAI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// OpenAI API key
    pub api_key: String,

    /// Model to use for chat completions
    #[serde(default = "default_model")]
    pub model: String,

    /// Model for embeddings
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// Maximum tokens for completions
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

/// Configuration for individual agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsConfig {
    /// Fulfillment agent configuration
    #[serde(default)]
    pub fulfillment: AgentInstanceConfig,

    /// Customer service agent configuration
    #[serde(default)]
    pub customer_service: AgentInstanceConfig,

    /// Supplier management agent configuration
    #[serde(default)]
    pub supplier: AgentInstanceConfig,

    /// Demand forecasting agent configuration
    #[serde(default)]
    pub demand_forecast: AgentInstanceConfig,

    /// Quality control agent configuration
    #[serde(default)]
    pub quality_control: AgentInstanceConfig,

    /// Order routing agent configuration
    #[serde(default)]
    pub order_routing: AgentInstanceConfig,
}

/// Configuration for a single agent instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInstanceConfig {
    /// Whether this agent is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Polling interval in seconds
    #[serde(default = "default_interval")]
    pub interval_secs: u64,

    /// Maximum concurrent tasks
    #[serde(default = "default_concurrency")]
    pub max_concurrency: usize,

    /// Custom system prompt override
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Confidence threshold for autonomous actions (0.0 - 1.0)
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,

    /// Whether to require human approval for high-impact actions
    #[serde(default = "default_true")]
    pub require_approval_for_high_impact: bool,
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,

    /// Key prefix for namespacing
    #[serde(default = "default_redis_prefix")]
    pub key_prefix: String,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Whether to output JSON format
    #[serde(default)]
    pub json: bool,
}

// Default value functions
fn default_timeout() -> u64 { 30 }
fn default_retries() -> u32 { 3 }
fn default_model() -> String { "gpt-4o-mini".to_string() }
fn default_embedding_model() -> String { "text-embedding-3-small".to_string() }
fn default_max_tokens() -> u32 { 2048 }
fn default_temperature() -> f32 { 0.7 }
fn default_true() -> bool { true }
fn default_interval() -> u64 { 60 }
fn default_concurrency() -> usize { 5 }
fn default_confidence_threshold() -> f32 { 0.8 }
fn default_redis_prefix() -> String { "stateset_agents:".to_string() }
fn default_log_level() -> String { "info".to_string() }

impl Default for AgentInstanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 60,
            max_concurrency: 5,
            system_prompt: None,
            confidence_threshold: 0.8,
            require_approval_for_high_impact: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json: false,
        }
    }
}

impl Default for AgentsConfig {
    fn default() -> Self {
        Self {
            fulfillment: AgentInstanceConfig::default(),
            customer_service: AgentInstanceConfig::default(),
            supplier: AgentInstanceConfig::default(),
            demand_forecast: AgentInstanceConfig { interval_secs: 3600, ..Default::default() },
            quality_control: AgentInstanceConfig::default(),
            order_routing: AgentInstanceConfig { interval_secs: 10, ..Default::default() },
        }
    }
}

impl AgentConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            api: ApiConfig {
                rest_url: std::env::var("STATESET_API_URL")
                    .unwrap_or_else(|_| "http://localhost:8080/api/v1".to_string()),
                grpc_url: std::env::var("STATESET_GRPC_URL").ok(),
                api_key: std::env::var("STATESET_API_KEY")
                    .map_err(|_| anyhow::anyhow!("STATESET_API_KEY environment variable required"))?,
                timeout_secs: std::env::var("STATESET_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
                max_retries: std::env::var("STATESET_MAX_RETRIES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3),
                prefer_grpc: std::env::var("STATESET_PREFER_GRPC")
                    .ok()
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false),
            },
            openai: OpenAIConfig {
                api_key: std::env::var("OPENAI_API_KEY")
                    .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable required"))?,
                model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| default_model()),
                embedding_model: std::env::var("OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| default_embedding_model()),
                max_tokens: std::env::var("OPENAI_MAX_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2048),
                temperature: std::env::var("OPENAI_TEMPERATURE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.7),
            },
            agents: AgentsConfig::default(),
            redis: std::env::var("REDIS_URL").ok().map(|url| RedisConfig {
                url,
                key_prefix: std::env::var("REDIS_PREFIX").unwrap_or_else(|_| default_redis_prefix()),
            }),
            logging: LoggingConfig {
                level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| default_log_level()),
                json: std::env::var("LOG_JSON").map(|v| v == "true").unwrap_or(false),
            },
        })
    }

    /// Get timeout as Duration
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.api.timeout_secs)
    }
}
