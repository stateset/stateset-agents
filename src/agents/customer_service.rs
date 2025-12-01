//! Customer Service Agent
//!
//! AI-powered customer service automation:
//! - Order status inquiries
//! - Return/refund processing
//! - Issue resolution
//! - Proactive customer outreach

use super::base::{Agent, AgentContext, AgentDecision, AgentEvent, BaseAgent};
use crate::clients::{
    CommerceApi, CreateReturnRequest, Customer, ListOrdersParams, ListReturnsParams,
    Order, OrderStatus, Return, ReturnItemRequest, ReturnStatus, StateSetClient,
};
use crate::config::AgentInstanceConfig;
use crate::services::LLMService;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Customer service automation agent
pub struct CustomerServiceAgent {
    base: BaseAgent,
}

impl CustomerServiceAgent {
    pub fn new(
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            base: BaseAgent::new("CustomerServiceAgent", client, llm, event_tx, config),
        }
    }

    /// Process a customer inquiry using AI
    pub async fn handle_inquiry(&self, inquiry: CustomerInquiry) -> anyhow::Result<InquiryResponse> {
        // Gather context about the customer
        let customer = self.base.client.get_customer(&inquiry.customer_id).await?;
        let orders = self.base.client.get_customer_orders(&inquiry.customer_id).await?;

        let context = self.build_customer_context(&customer, &orders.items);

        let system_prompt = self.get_customer_service_prompt();
        let user_prompt = format!(
            "Customer Context:\n{}\n\nCustomer Inquiry:\n{}",
            context, inquiry.message
        );

        let response = self.base.llm.chat_completion(&system_prompt, &user_prompt).await?;

        let analysis: InquiryAnalysis = self.base.parse_llm_json(&response)?;

        // Execute recommended actions if confidence is high
        if analysis.confidence >= self.base.config.confidence_threshold {
            for action in &analysis.actions {
                if let Err(e) = self.execute_action(action, &inquiry).await {
                    warn!("Failed to execute action {:?}: {}", action, e);
                }
            }
        }

        let decision = AgentDecision::new(
            "customer_inquiry",
            &analysis.intent,
            analysis.confidence,
            &analysis.response,
        ).with_metadata(serde_json::json!({
            "customer_id": inquiry.customer_id,
            "actions": analysis.actions,
        }));

        self.base.emit_decision(&decision).await;

        Ok(InquiryResponse {
            message: analysis.response,
            actions_taken: analysis.actions,
            requires_escalation: analysis.requires_escalation,
            sentiment: analysis.sentiment,
        })
    }

    /// Build context string for customer service
    fn build_customer_context(&self, customer: &Customer, orders: &[Order]) -> String {
        let recent_orders: Vec<_> = orders.iter().take(5).collect();

        format!(
            r#"Customer: {} {}
Email: {}
Total Orders: {}
Recent Orders:
{}"#,
            customer.first_name.as_deref().unwrap_or(""),
            customer.last_name.as_deref().unwrap_or(""),
            customer.email,
            orders.len(),
            recent_orders.iter()
                .map(|o| format!("  - {} ({}): ${:.2} - {}",
                    o.order_number.as_deref().unwrap_or(&o.id),
                    o.created_at,
                    o.total_amount,
                    format!("{:?}", o.status)
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Get the system prompt for customer service
    fn get_customer_service_prompt(&self) -> String {
        self.base.config.system_prompt.clone().unwrap_or_else(|| {
            r#"You are an AI Customer Service Agent for StateSet Commerce. Your role is to:
1. Understand customer inquiries and intent
2. Provide helpful, accurate responses
3. Take appropriate actions when authorized
4. Escalate complex issues to human agents

Available Actions:
- CHECK_ORDER_STATUS: Look up order details
- INITIATE_RETURN: Start a return process
- APPLY_CREDIT: Apply store credit (up to $25 for minor issues)
- UPDATE_ADDRESS: Update shipping address
- CANCEL_ORDER: Cancel an unshipped order
- ESCALATE: Transfer to human agent

Respond with JSON:
{
    "intent": "identified customer intent",
    "sentiment": "positive/neutral/negative",
    "response": "customer-facing response message",
    "actions": ["ACTION_1", "ACTION_2"],
    "requires_escalation": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "internal reasoning"
}"#.to_string()
        })
    }

    /// Execute a customer service action
    async fn execute_action(&self, action: &str, inquiry: &CustomerInquiry) -> anyhow::Result<()> {
        match action {
            "CHECK_ORDER_STATUS" => {
                if let Some(order_id) = &inquiry.order_id {
                    let order = self.base.client.get_order(order_id).await?;
                    info!("Checked order {} status: {:?}", order_id, order.status);
                }
            }
            "INITIATE_RETURN" => {
                if let Some(order_id) = &inquiry.order_id {
                    let order = self.base.client.get_order(order_id).await?;

                    let return_req = self.base.client.create_return(CreateReturnRequest {
                        order_id: order_id.clone(),
                        items: order.items.iter().map(|i| ReturnItemRequest {
                            order_item_id: i.id.clone(),
                            quantity: i.quantity,
                            reason: inquiry.message.clone(),
                        }).collect(),
                        reason: "Customer requested return".to_string(),
                        customer_notes: Some(inquiry.message.clone()),
                    }).await?;

                    info!("Created return {} for order {}", return_req.id, order_id);
                    self.base.emit_action("initiate_return", Ok(return_req.id)).await;
                }
            }
            "APPLY_CREDIT" => {
                // Would integrate with credits/loyalty system
                info!("Would apply credit to customer {}", inquiry.customer_id);
            }
            "CANCEL_ORDER" => {
                if let Some(order_id) = &inquiry.order_id {
                    let order = self.base.client.get_order(order_id).await?;

                    // Only cancel if not yet shipped
                    if order.status == OrderStatus::Pending || order.status == OrderStatus::Processing {
                        self.base.client.cancel_order(order_id, "Customer requested cancellation").await?;
                        info!("Cancelled order {}", order_id);
                        self.base.emit_action("cancel_order", Ok(order_id.clone())).await;
                    } else {
                        warn!("Cannot cancel order {} - status is {:?}", order_id, order.status);
                    }
                }
            }
            "ESCALATE" => {
                self.base.emit_alert("warning", &format!(
                    "Customer {} inquiry requires escalation: {}",
                    inquiry.customer_id, inquiry.message
                )).await;
            }
            _ => {
                warn!("Unknown action: {}", action);
            }
        }

        Ok(())
    }

    /// Proactively reach out to customers with issues
    async fn proactive_outreach(&self) -> anyhow::Result<()> {
        // Find orders with delivery issues
        let orders = self.base.client.list_orders(ListOrdersParams {
            status: Some(OrderStatus::Shipped),
            limit: Some(100),
            ..Default::default()
        }).await?;

        // Find delayed shipments (simplified check)
        for order in orders.items {
            // In production, would check actual delivery estimates
            let days_since_shipped = 5; // Placeholder

            if days_since_shipped > 7 {
                info!("Order {} may be delayed, considering proactive outreach", order.id);

                let decision = AgentDecision::new(
                    "proactive_outreach",
                    "send_delay_notification",
                    0.85,
                    "Order appears delayed beyond expected delivery window",
                ).with_metadata(serde_json::json!({
                    "order_id": order.id,
                    "customer_id": order.customer_id,
                }));

                self.base.emit_decision(&decision).await;
            }
        }

        Ok(())
    }

    /// Analyze return patterns for quality insights
    async fn analyze_return_patterns(&self) -> anyhow::Result<()> {
        let returns = self.base.client.list_returns(ListReturnsParams {
            limit: Some(100),
            ..Default::default()
        }).await?;

        // Group by reason
        let mut reason_counts: HashMap<String, i32> = HashMap::new();
        for ret in &returns.items {
            *reason_counts.entry(ret.reason.clone()).or_insert(0) += 1;
        }

        // Check for patterns
        let total = returns.items.len() as f64;
        for (reason, count) in &reason_counts {
            let percentage = (*count as f64 / total) * 100.0;
            if percentage > 20.0 {
                self.base.emit_alert(
                    "warning",
                    &format!("High return rate for reason '{}': {:.1}%", reason, percentage),
                ).await;
            }
        }

        self.base.emit_metric("total_returns_analyzed", returns.items.len() as f64).await;

        Ok(())
    }
}

#[async_trait]
impl Agent for CustomerServiceAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        "Provides AI-powered customer service automation and support"
    }

    async fn run(&self) {
        let interval = Duration::from_secs(self.base.config.interval_secs);
        let mut ticker = tokio::time::interval(interval);

        info!("{} started, running every {:?}", self.name(), interval);

        loop {
            ticker.tick().await;

            let ctx = AgentContext::new(self.base.config.clone());

            if let Err(e) = self.process_cycle(&ctx).await {
                error!("{} cycle error: {}", self.name(), e);
            }
        }
    }

    async fn process_cycle(&self, _ctx: &AgentContext) -> anyhow::Result<Vec<AgentDecision>> {
        let mut decisions = Vec::new();

        // Proactive customer outreach
        if let Err(e) = self.proactive_outreach().await {
            warn!("Proactive outreach failed: {}", e);
        }

        // Analyze return patterns
        if let Err(e) = self.analyze_return_patterns().await {
            warn!("Return analysis failed: {}", e);
        }

        Ok(decisions)
    }

    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()> {
        info!("Executing customer service decision: {}", decision.action);
        Ok(())
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Customer inquiry to be processed
#[derive(Debug, Clone)]
pub struct CustomerInquiry {
    pub customer_id: String,
    pub message: String,
    pub order_id: Option<String>,
    pub channel: String, // email, chat, phone
}

/// Response to a customer inquiry
#[derive(Debug, Clone, Serialize)]
pub struct InquiryResponse {
    pub message: String,
    pub actions_taken: Vec<String>,
    pub requires_escalation: bool,
    pub sentiment: String,
}

#[derive(Debug, Deserialize)]
struct InquiryAnalysis {
    intent: String,
    sentiment: String,
    response: String,
    actions: Vec<String>,
    requires_escalation: bool,
    confidence: f32,
    #[allow(dead_code)]
    reasoning: String,
}
