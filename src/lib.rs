//! StateSet Agents - AI Agents Framework for Intelligent Commerce
//!
//! This library provides autonomous AI agents that integrate with the StateSet API
//! to automate and optimize commerce operations.
//!
//! # Features
//!
//! - **Unified API Client**: REST and gRPC clients with automatic failover
//! - **AI-Powered Agents**: Autonomous agents for fulfillment, customer service, and more
//! - **Event-Driven**: Agents emit events for monitoring and coordination
//! - **Configurable**: Full configuration via environment variables or config files
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use stateset_agents::{AgentConfig, AgentManager};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load config from environment
//!     let config = AgentConfig::from_env()?;
//!
//!     // Create and start agent manager
//!     let mut manager = AgentManager::new(config).await?;
//!     manager.start_all().await?;
//!
//!     // Keep running
//!     tokio::signal::ctrl_c().await?;
//!     Ok(())
//! }
//! ```
//!
//! # Available Agents
//!
//! - **FulfillmentAgent**: Optimizes order fulfillment and shipping
//! - **CustomerServiceAgent**: Handles customer inquiries with AI
//! - **SupplierManagementAgent**: Manages procurement and supplier relationships
//! - **DemandForecastAgent**: Predicts demand and optimizes inventory
//! - **QualityControlAgent**: Monitors quality through return analysis
//! - **OrderRoutingAgent**: Routes orders to optimal fulfillment locations
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StateSet Agents                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │  │ Fulfillment │  │  Customer   │  │  Supplier   │  ...   │
//! │  │   Agent     │  │   Service   │  │   Agent     │        │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
//! │         │                │                │                │
//! │  ┌──────┴────────────────┴────────────────┴──────┐        │
//! │  │              Agent Manager                     │        │
//! │  │         (Orchestration & Events)              │        │
//! │  └──────────────────┬────────────────────────────┘        │
//! │                     │                                      │
//! │  ┌──────────────────┴────────────────────────────┐        │
//! │  │           Unified API Client                  │        │
//! │  │         (REST + gRPC)                         │        │
//! │  └──────────────────┬────────────────────────────┘        │
//! └─────────────────────┼─────────────────────────────────────┘
//!                       │
//!                       ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   StateSet API                              │
//! │   (Orders, Inventory, Returns, Shipments, Analytics)        │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod agents;
pub mod clients;
pub mod config;
pub mod services;

// Re-exports for convenience
pub use agents::{
    Agent, AgentContext, AgentDecision, AgentEvent, AgentManager,
    CustomerServiceAgent, DemandForecastAgent, FulfillmentAgent,
    OrderRoutingAgent, QualityControlAgent, SupplierManagementAgent,
};
pub use clients::{CommerceApi, RestClient, StateSetClient};
pub use config::{AgentConfig, AgentInstanceConfig, ApiConfig, OpenAIConfig};
pub use services::LLMService;

/// Version of the stateset-agents library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
