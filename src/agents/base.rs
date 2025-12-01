//! Base Agent Traits and Types
//!
//! Provides the foundational types and traits for all AI agents.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::clients::StateSetClient;
use crate::config::AgentInstanceConfig;
use crate::services::LLMService;

/// Context provided to agents for making decisions
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Current timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Agent configuration
    pub config: AgentInstanceConfig,
    /// Optional correlation ID for tracing
    pub correlation_id: Option<String>,
}

impl AgentContext {
    pub fn new(config: AgentInstanceConfig) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            config,
            correlation_id: Some(uuid::Uuid::new_v4().to_string()),
        }
    }
}

/// Represents a decision made by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDecision {
    /// Type of decision
    pub decision_type: String,
    /// The chosen action
    pub action: String,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Reasoning behind the decision
    pub reasoning: String,
    /// Whether this requires human approval
    pub requires_approval: bool,
    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl AgentDecision {
    pub fn new(decision_type: &str, action: &str, confidence: f32, reasoning: &str) -> Self {
        Self {
            decision_type: decision_type.to_string(),
            action: action.to_string(),
            confidence,
            reasoning: reasoning.to_string(),
            requires_approval: false,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_approval(mut self, requires: bool) -> Self {
        self.requires_approval = requires;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Events emitted by agents for monitoring and coordination
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Agent made a decision
    DecisionMade {
        agent: String,
        decision: AgentDecision,
        confidence: f32,
    },
    /// Agent took an action
    ActionTaken {
        agent: String,
        action: String,
        result: Result<String, String>,
    },
    /// Agent raised an alert
    Alert {
        agent: String,
        severity: String,
        message: String,
    },
    /// Agent recorded a metric
    MetricRecorded {
        agent: String,
        metric: String,
        value: f64,
    },
}

/// Base trait for all agents
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get the agent's name
    fn name(&self) -> &str;

    /// Get the agent's description
    fn description(&self) -> &str;

    /// Run the agent's main loop
    async fn run(&self);

    /// Process a single cycle
    async fn process_cycle(&self, ctx: &AgentContext) -> anyhow::Result<Vec<AgentDecision>>;

    /// Execute a decision
    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()>;

    /// Check if the agent should process (rate limiting, etc.)
    fn should_process(&self, ctx: &AgentContext) -> bool {
        true
    }
}

/// Common agent implementation with shared functionality
pub struct BaseAgent {
    pub name: String,
    pub client: Arc<StateSetClient>,
    pub llm: Arc<LLMService>,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub config: AgentInstanceConfig,
}

impl BaseAgent {
    pub fn new(
        name: &str,
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            name: name.to_string(),
            client,
            llm,
            event_tx,
            config,
        }
    }

    /// Emit an event
    pub async fn emit_event(&self, event: AgentEvent) {
        if let Err(e) = self.event_tx.send(event).await {
            tracing::error!("Failed to emit event: {}", e);
        }
    }

    /// Emit a decision event
    pub async fn emit_decision(&self, decision: &AgentDecision) {
        self.emit_event(AgentEvent::DecisionMade {
            agent: self.name.clone(),
            decision: decision.clone(),
            confidence: decision.confidence,
        }).await;
    }

    /// Emit an action event
    pub async fn emit_action(&self, action: &str, result: Result<String, String>) {
        self.emit_event(AgentEvent::ActionTaken {
            agent: self.name.clone(),
            action: action.to_string(),
            result,
        }).await;
    }

    /// Emit an alert
    pub async fn emit_alert(&self, severity: &str, message: &str) {
        self.emit_event(AgentEvent::Alert {
            agent: self.name.clone(),
            severity: severity.to_string(),
            message: message.to_string(),
        }).await;
    }

    /// Emit a metric
    pub async fn emit_metric(&self, metric: &str, value: f64) {
        self.emit_event(AgentEvent::MetricRecorded {
            agent: self.name.clone(),
            metric: metric.to_string(),
            value,
        }).await;
    }

    /// Check if decision meets confidence threshold
    pub fn meets_threshold(&self, decision: &AgentDecision) -> bool {
        decision.confidence >= self.config.confidence_threshold
    }

    /// Check if decision requires approval based on config
    pub fn requires_approval(&self, decision: &AgentDecision) -> bool {
        decision.requires_approval ||
        (self.config.require_approval_for_high_impact && decision.confidence < 0.9)
    }

    /// Parse LLM JSON response
    pub fn parse_llm_json<T: serde::de::DeserializeOwned>(&self, response: &str) -> anyhow::Result<T> {
        // Try to extract JSON from markdown code blocks
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };

        serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse LLM response: {} - Response: {}", e, response))
    }
}

/// LLM prompt templates for agents
pub mod prompts {
    /// Create a decision-making prompt
    pub fn decision_prompt(context: &str, options: &[&str]) -> String {
        format!(
            r#"You are an AI commerce agent. Analyze the following context and make a decision.

Context:
{}

Available Actions:
{}

Respond with a JSON object:
{{
    "action": "chosen_action",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "requires_human_review": true/false
}}"#,
            context,
            options.iter().enumerate()
                .map(|(i, o)| format!("{}. {}", i + 1, o))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Create an analysis prompt
    pub fn analysis_prompt(data: &str, question: &str) -> String {
        format!(
            r#"You are an AI commerce analyst. Analyze the following data and answer the question.

Data:
{}

Question: {}

Provide a detailed analysis in JSON format with:
- key_findings: list of important observations
- recommendations: list of suggested actions
- risk_level: low/medium/high
- confidence: 0.0-1.0"#,
            data, question
        )
    }
}
