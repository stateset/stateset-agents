//! Fulfillment Agent
//!
//! Autonomous agent for optimizing order fulfillment operations:
//! - Optimal warehouse selection
//! - Carrier selection and rate shopping
//! - Batch shipment creation
//! - Delivery exception handling

use super::base::{Agent, AgentContext, AgentDecision, AgentEvent, BaseAgent};
use crate::clients::{
    CommerceApi, CreateShipmentRequest, ListOrdersParams, Order, OrderStatus,
    ShipmentItemRequest, ShipmentStatus, StateSetClient,
};
use crate::config::AgentInstanceConfig;
use crate::services::LLMService;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Fulfillment optimization agent
pub struct FulfillmentAgent {
    base: BaseAgent,
}

impl FulfillmentAgent {
    pub fn new(
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            base: BaseAgent::new("FulfillmentAgent", client, llm, event_tx, config),
        }
    }

    /// Analyze pending orders and create optimal fulfillment plan
    async fn analyze_pending_orders(&self) -> anyhow::Result<Vec<FulfillmentPlan>> {
        let orders = self.base.client.list_orders(ListOrdersParams {
            status: Some(OrderStatus::Processing),
            limit: Some(100),
            ..Default::default()
        }).await?;

        let mut plans = Vec::new();

        for order in orders.items {
            if let Some(plan) = self.create_fulfillment_plan(&order).await? {
                plans.push(plan);
            }
        }

        Ok(plans)
    }

    /// Create an optimal fulfillment plan for an order
    async fn create_fulfillment_plan(&self, order: &Order) -> anyhow::Result<Option<FulfillmentPlan>> {
        // Build context for LLM
        let order_json = serde_json::to_string_pretty(order)?;

        let system_prompt = r#"You are a Fulfillment Optimization Agent. Analyze the order and determine the optimal fulfillment strategy.

Consider:
1. Shipping destination and available warehouses
2. Item availability across locations
3. Shipping costs and delivery time requirements
4. Order priority and customer tier

Respond with JSON:
{
    "warehouse_id": "optimal warehouse",
    "carrier": "recommended carrier (UPS/FedEx/USPS/DHL)",
    "service_level": "ground/express/overnight",
    "estimated_cost": 0.00,
    "estimated_days": 0,
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}"#;

        let user_prompt = format!("Order to fulfill:\n{}", order_json);

        let response = self.base.llm.chat_completion(system_prompt, &user_prompt).await?;

        let analysis: FulfillmentAnalysis = self.base.parse_llm_json(&response)?;

        if analysis.confidence < self.base.config.confidence_threshold {
            info!(
                "Skipping order {} - confidence {:.2} below threshold",
                order.id, analysis.confidence
            );
            return Ok(None);
        }

        Ok(Some(FulfillmentPlan {
            order_id: order.id.clone(),
            warehouse_id: analysis.warehouse_id,
            carrier: analysis.carrier,
            service_level: analysis.service_level,
            items: order.items.iter().map(|i| FulfillmentItem {
                order_item_id: i.id.clone(),
                quantity: i.quantity,
            }).collect(),
            estimated_cost: analysis.estimated_cost,
            estimated_days: analysis.estimated_days,
            confidence: analysis.confidence,
            reasoning: analysis.reasoning,
        }))
    }

    /// Execute a fulfillment plan by creating a shipment
    async fn execute_plan(&self, plan: &FulfillmentPlan) -> anyhow::Result<()> {
        info!(
            "Executing fulfillment plan for order {} via {} {}",
            plan.order_id, plan.carrier, plan.service_level
        );

        // Create shipment via API
        let shipment = self.base.client.create_shipment(CreateShipmentRequest {
            order_id: plan.order_id.clone(),
            carrier: Some(plan.carrier.clone()),
            service_level: Some(plan.service_level.clone()),
            items: plan.items.iter().map(|i| ShipmentItemRequest {
                order_item_id: i.order_item_id.clone(),
                quantity: i.quantity,
            }).collect(),
        }).await?;

        info!("Created shipment {} for order {}", shipment.id, plan.order_id);

        // Update order status
        self.base.client.update_order_status(
            &plan.order_id,
            OrderStatus::Shipped,
            Some(format!("Shipped via {} {}", plan.carrier, plan.service_level)),
        ).await?;

        self.base.emit_action(
            "create_shipment",
            Ok(format!("Shipment {} created", shipment.id)),
        ).await;

        self.base.emit_metric("shipment_created", 1.0).await;
        self.base.emit_metric("estimated_shipping_cost", plan.estimated_cost).await;

        Ok(())
    }

    /// Handle delivery exceptions
    async fn handle_exceptions(&self) -> anyhow::Result<()> {
        let shipments = self.base.client.list_shipments(crate::clients::ListShipmentsParams {
            status: Some(ShipmentStatus::Failed),
            limit: Some(50),
            ..Default::default()
        }).await?;

        for shipment in shipments.items {
            info!("Analyzing failed shipment {}", shipment.id);

            let system_prompt = r#"You are a Delivery Exception Handler. Analyze the failed shipment and recommend recovery action.

Options:
1. RETRY - Attempt redelivery with same carrier
2. REROUTE - Ship via different carrier
3. PICKUP - Arrange customer pickup
4. REFUND - Process refund and cancel
5. ESCALATE - Requires human intervention

Respond with JSON:
{
    "action": "ACTION_TYPE",
    "reasoning": "explanation",
    "confidence": 0.0-1.0
}"#;

            let shipment_json = serde_json::to_string_pretty(&shipment)?;
            let response = self.base.llm.chat_completion(system_prompt, &shipment_json).await?;

            #[derive(Deserialize)]
            struct ExceptionAction {
                action: String,
                reasoning: String,
                confidence: f32,
            }

            if let Ok(action) = self.base.parse_llm_json::<ExceptionAction>(&response) {
                let decision = AgentDecision::new(
                    "delivery_exception",
                    &action.action,
                    action.confidence,
                    &action.reasoning,
                ).with_approval(action.action == "REFUND" || action.action == "ESCALATE");

                self.base.emit_decision(&decision).await;

                if self.base.meets_threshold(&decision) && !self.base.requires_approval(&decision) {
                    match action.action.as_str() {
                        "RETRY" => {
                            info!("Retrying shipment {}", shipment.id);
                            // Would trigger redelivery logic
                        }
                        "REROUTE" => {
                            info!("Rerouting shipment {} to alternate carrier", shipment.id);
                            // Would create new shipment with different carrier
                        }
                        _ => {
                            warn!("Action {} requires manual handling", action.action);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Agent for FulfillmentAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        "Optimizes order fulfillment through intelligent warehouse and carrier selection"
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
                self.base.emit_alert("error", &format!("Cycle failed: {}", e)).await;
            }
        }
    }

    async fn process_cycle(&self, ctx: &AgentContext) -> anyhow::Result<Vec<AgentDecision>> {
        let mut decisions = Vec::new();

        // 1. Analyze and plan fulfillment for pending orders
        let plans = self.analyze_pending_orders().await?;
        info!("{} created {} fulfillment plans", self.name(), plans.len());

        for plan in &plans {
            let decision = AgentDecision::new(
                "fulfillment_plan",
                &format!("ship_via_{}_{}", plan.carrier, plan.service_level),
                plan.confidence,
                &plan.reasoning,
            ).with_metadata(serde_json::json!({
                "order_id": plan.order_id,
                "warehouse_id": plan.warehouse_id,
                "estimated_cost": plan.estimated_cost,
                "estimated_days": plan.estimated_days,
            }));

            decisions.push(decision);
        }

        // 2. Execute plans that meet threshold
        for plan in plans {
            if plan.confidence >= self.base.config.confidence_threshold {
                if let Err(e) = self.execute_plan(&plan).await {
                    error!("Failed to execute plan for order {}: {}", plan.order_id, e);
                }
            }
        }

        // 3. Handle delivery exceptions
        if let Err(e) = self.handle_exceptions().await {
            warn!("Exception handling failed: {}", e);
        }

        Ok(decisions)
    }

    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()> {
        // Decision execution is handled in process_cycle for this agent
        Ok(())
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FulfillmentAnalysis {
    warehouse_id: String,
    carrier: String,
    service_level: String,
    estimated_cost: f64,
    estimated_days: i32,
    confidence: f32,
    reasoning: String,
}

#[derive(Debug, Clone)]
struct FulfillmentPlan {
    order_id: String,
    warehouse_id: String,
    carrier: String,
    service_level: String,
    items: Vec<FulfillmentItem>,
    estimated_cost: f64,
    estimated_days: i32,
    confidence: f32,
    reasoning: String,
}

#[derive(Debug, Clone)]
struct FulfillmentItem {
    order_item_id: String,
    quantity: i32,
}
