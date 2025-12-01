//! AI Agents for Intelligent Commerce
//!
//! This module provides autonomous AI agents that integrate with the StateSet API
//! to automate commerce operations.

pub mod base;
pub mod fulfillment;
pub mod customer_service;
pub mod supplier;
pub mod demand_forecast;
pub mod quality_control;
pub mod order_routing;

pub use base::{Agent, AgentContext, AgentDecision, AgentEvent};
pub use fulfillment::FulfillmentAgent;
pub use customer_service::CustomerServiceAgent;
pub use supplier::SupplierManagementAgent;
pub use demand_forecast::DemandForecastAgent;
pub use quality_control::QualityControlAgent;
pub use order_routing::OrderRoutingAgent;

use crate::clients::StateSetClient;
use crate::config::AgentConfig;
use crate::services::LLMService;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

/// Agent manager that orchestrates all agents
pub struct AgentManager {
    config: AgentConfig,
    client: Arc<StateSetClient>,
    llm: Arc<LLMService>,
    event_tx: mpsc::Sender<AgentEvent>,
    event_rx: Option<mpsc::Receiver<AgentEvent>>,
}

impl AgentManager {
    /// Create a new agent manager
    pub async fn new(config: AgentConfig) -> anyhow::Result<Self> {
        let client = Arc::new(StateSetClient::new(&config).await?);
        let llm = Arc::new(LLMService::new(&config.openai)?);
        let (event_tx, event_rx) = mpsc::channel(1024);

        Ok(Self {
            config,
            client,
            llm,
            event_tx,
            event_rx: Some(event_rx),
        })
    }

    /// Start all enabled agents
    pub async fn start_all(&mut self) -> anyhow::Result<()> {
        let mut handles = Vec::new();

        // Fulfillment Agent
        if self.config.agents.fulfillment.enabled {
            let agent = FulfillmentAgent::new(
                self.client.clone(),
                self.llm.clone(),
                self.event_tx.clone(),
                self.config.agents.fulfillment.clone(),
            );
            let handle = tokio::spawn(async move {
                agent.run().await;
            });
            handles.push(handle);
            info!("Started FulfillmentAgent");
        }

        // Customer Service Agent
        if self.config.agents.customer_service.enabled {
            let agent = CustomerServiceAgent::new(
                self.client.clone(),
                self.llm.clone(),
                self.event_tx.clone(),
                self.config.agents.customer_service.clone(),
            );
            let handle = tokio::spawn(async move {
                agent.run().await;
            });
            handles.push(handle);
            info!("Started CustomerServiceAgent");
        }

        // Supplier Management Agent
        if self.config.agents.supplier.enabled {
            let agent = SupplierManagementAgent::new(
                self.client.clone(),
                self.llm.clone(),
                self.event_tx.clone(),
                self.config.agents.supplier.clone(),
            );
            let handle = tokio::spawn(async move {
                agent.run().await;
            });
            handles.push(handle);
            info!("Started SupplierManagementAgent");
        }

        // Demand Forecast Agent
        if self.config.agents.demand_forecast.enabled {
            let agent = DemandForecastAgent::new(
                self.client.clone(),
                self.llm.clone(),
                self.event_tx.clone(),
                self.config.agents.demand_forecast.clone(),
            );
            let handle = tokio::spawn(async move {
                agent.run().await;
            });
            handles.push(handle);
            info!("Started DemandForecastAgent");
        }

        // Quality Control Agent
        if self.config.agents.quality_control.enabled {
            let agent = QualityControlAgent::new(
                self.client.clone(),
                self.llm.clone(),
                self.event_tx.clone(),
                self.config.agents.quality_control.clone(),
            );
            let handle = tokio::spawn(async move {
                agent.run().await;
            });
            handles.push(handle);
            info!("Started QualityControlAgent");
        }

        // Order Routing Agent
        if self.config.agents.order_routing.enabled {
            let agent = OrderRoutingAgent::new(
                self.client.clone(),
                self.llm.clone(),
                self.event_tx.clone(),
                self.config.agents.order_routing.clone(),
            );
            let handle = tokio::spawn(async move {
                agent.run().await;
            });
            handles.push(handle);
            info!("Started OrderRoutingAgent");
        }

        // Start event processor
        if let Some(rx) = self.event_rx.take() {
            tokio::spawn(async move {
                process_agent_events(rx).await;
            });
        }

        info!("All agents started ({} total)", handles.len());
        Ok(())
    }

    /// Get the event sender for external event injection
    pub fn event_sender(&self) -> mpsc::Sender<AgentEvent> {
        self.event_tx.clone()
    }

    /// Get the API client
    pub fn client(&self) -> Arc<StateSetClient> {
        self.client.clone()
    }
}

/// Process events emitted by agents
async fn process_agent_events(mut rx: mpsc::Receiver<AgentEvent>) {
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::DecisionMade { agent, decision, confidence } => {
                info!(
                    "Agent '{}' made decision: {:?} (confidence: {:.2})",
                    agent, decision, confidence
                );
            }
            AgentEvent::ActionTaken { agent, action, result } => {
                info!(
                    "Agent '{}' took action '{}': {}",
                    agent, action, if result.is_ok() { "success" } else { "failed" }
                );
            }
            AgentEvent::Alert { agent, severity, message } => {
                match severity.as_str() {
                    "critical" => tracing::error!("[{}] ALERT: {}", agent, message),
                    "warning" => tracing::warn!("[{}] ALERT: {}", agent, message),
                    _ => tracing::info!("[{}] ALERT: {}", agent, message),
                }
            }
            AgentEvent::MetricRecorded { agent, metric, value } => {
                tracing::debug!("[{}] Metric '{}': {}", agent, metric, value);
            }
        }
    }
}
