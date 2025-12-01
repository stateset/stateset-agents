//! Order Routing Agent
//!
//! Intelligent order routing and orchestration:
//! - Optimal fulfillment location selection
//! - Split shipment decisions
//! - Priority routing
//! - Real-time inventory allocation

use super::base::{Agent, AgentContext, AgentDecision, AgentEvent, BaseAgent};
use crate::clients::{
    CommerceApi, InventoryItem, ListInventoryParams, ListOrdersParams, Order, OrderStatus,
    ReserveInventoryRequest, StateSetClient,
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

/// Order routing agent
pub struct OrderRoutingAgent {
    base: BaseAgent,
}

impl OrderRoutingAgent {
    pub fn new(
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            base: BaseAgent::new("OrderRoutingAgent", client, llm, event_tx, config),
        }
    }

    /// Route pending orders to optimal fulfillment locations
    async fn route_pending_orders(&self) -> anyhow::Result<Vec<RoutingDecision>> {
        let orders = self.base.client.list_orders(ListOrdersParams {
            status: Some(OrderStatus::Pending),
            limit: Some(50),
            ..Default::default()
        }).await?;

        let mut routing_decisions = Vec::new();

        for order in orders.items {
            if let Some(decision) = self.route_order(&order).await? {
                routing_decisions.push(decision);
            }
        }

        Ok(routing_decisions)
    }

    /// Route a single order
    async fn route_order(&self, order: &Order) -> anyhow::Result<Option<RoutingDecision>> {
        // Get inventory availability for all items
        let mut inventory_by_item: HashMap<String, Vec<InventoryItem>> = HashMap::new();

        for item in &order.items {
            let inventory = self.base.client.list_inventory(ListInventoryParams {
                product_id: Some(item.product_id.clone()),
                limit: Some(100),
                ..Default::default()
            }).await?;

            inventory_by_item.insert(item.product_id.clone(), inventory.items);
        }

        // Build context for AI routing decision
        let order_json = serde_json::to_string_pretty(order)?;
        let inventory_json = serde_json::to_string_pretty(&inventory_by_item)?;

        let shipping_address = order.shipping_address.as_ref()
            .map(|a| format!("{}, {}", a.city, a.country))
            .unwrap_or_else(|| "Unknown".to_string());

        let context = format!(
            r#"Order Details:
{}

Inventory Availability by Location:
{}

Shipping Destination: {}"#,
            order_json, inventory_json, shipping_address
        );

        let system_prompt = r#"You are an Order Routing AI. Determine the optimal fulfillment strategy for this order.

Consider:
1. Inventory availability across locations
2. Distance to shipping destination
3. Shipping costs and speed
4. Whether to split the shipment

Respond with JSON:
{
    "strategy": "single_location/split_shipment/backorder",
    "primary_location": "location_id or null",
    "allocations": [
        {"product_id": "...", "location_id": "...", "quantity": 0}
    ],
    "split_reason": "reason if split, null otherwise",
    "estimated_ship_date": "YYYY-MM-DD",
    "priority": "standard/expedited/next_day",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}"#;

        let response = self.base.llm.chat_completion(system_prompt, &context).await?;

        if let Ok(routing) = self.base.parse_llm_json::<RoutingAnalysis>(&response) {
            if routing.confidence >= self.base.config.confidence_threshold {
                return Ok(Some(RoutingDecision {
                    order_id: order.id.clone(),
                    strategy: routing.strategy,
                    primary_location: routing.primary_location,
                    allocations: routing.allocations,
                    priority: routing.priority,
                    confidence: routing.confidence,
                    reasoning: routing.reasoning,
                }));
            }
        }

        Ok(None)
    }

    /// Execute routing decisions by allocating inventory
    async fn execute_routing(&self, decisions: Vec<RoutingDecision>) -> anyhow::Result<()> {
        for decision in decisions {
            info!(
                "Routing order {} via {} strategy",
                decision.order_id, decision.strategy
            );

            let agent_decision = AgentDecision::new(
                "order_routing",
                &decision.strategy,
                decision.confidence,
                &decision.reasoning,
            ).with_metadata(serde_json::json!({
                "order_id": decision.order_id,
                "allocations": decision.allocations,
                "priority": decision.priority,
            }));

            self.base.emit_decision(&agent_decision).await;

            // Reserve inventory for each allocation
            for allocation in &decision.allocations {
                // Find the inventory item ID for this product/location
                let inventory = self.base.client.list_inventory(ListInventoryParams {
                    product_id: Some(allocation.product_id.clone()),
                    location_id: Some(allocation.location_id.clone()),
                    ..Default::default()
                }).await?;

                if let Some(inv) = inventory.items.first() {
                    match self.base.client.reserve_inventory(ReserveInventoryRequest {
                        inventory_id: inv.id.clone(),
                        quantity: allocation.quantity,
                        order_id: Some(decision.order_id.clone()),
                        notes: Some(format!("Reserved by OrderRoutingAgent: {}", decision.reasoning)),
                    }).await {
                        Ok(reservation) => {
                            info!(
                                "Reserved {} units of {} at {} for order {}",
                                allocation.quantity, allocation.product_id,
                                allocation.location_id, decision.order_id
                            );
                            self.base.emit_action(
                                "reserve_inventory",
                                Ok(reservation.id),
                            ).await;
                        }
                        Err(e) => {
                            warn!(
                                "Failed to reserve inventory for order {}: {}",
                                decision.order_id, e
                            );
                            self.base.emit_action(
                                "reserve_inventory",
                                Err(e.to_string()),
                            ).await;
                        }
                    }
                }
            }

            // Update order status to processing
            if let Err(e) = self.base.client.update_order_status(
                &decision.order_id,
                OrderStatus::Processing,
                Some(format!("Routed via {} strategy", decision.strategy)),
            ).await {
                warn!("Failed to update order status: {}", e);
            }

            self.base.emit_metric("orders_routed", 1.0).await;
        }

        Ok(())
    }

    /// Handle orders that can't be fulfilled
    async fn handle_unfulfillable_orders(&self) -> anyhow::Result<()> {
        let orders = self.base.client.list_orders(ListOrdersParams {
            status: Some(OrderStatus::Pending),
            limit: Some(100),
            ..Default::default()
        }).await?;

        for order in orders.items {
            let mut can_fulfill = true;

            for item in &order.items {
                let inventory = self.base.client.list_inventory(ListInventoryParams {
                    product_id: Some(item.product_id.clone()),
                    ..Default::default()
                }).await?;

                let total_available: i32 = inventory.items.iter()
                    .map(|i| i.quantity_available)
                    .sum();

                if total_available < item.quantity {
                    can_fulfill = false;
                    break;
                }
            }

            if !can_fulfill {
                self.base.emit_alert(
                    "warning",
                    &format!("Order {} cannot be fulfilled - insufficient inventory", order.id),
                ).await;

                // Decide what to do with unfulfillable order
                let decision = AgentDecision::new(
                    "unfulfillable_order",
                    "notify_customer_backorder",
                    0.9,
                    "Order contains items that are out of stock across all locations",
                ).with_metadata(serde_json::json!({
                    "order_id": order.id,
                })).with_approval(true);

                self.base.emit_decision(&decision).await;
            }
        }

        Ok(())
    }

    /// Optimize batch routing for efficiency
    async fn optimize_batch_routing(&self) -> anyhow::Result<()> {
        let orders = self.base.client.list_orders(ListOrdersParams {
            status: Some(OrderStatus::Pending),
            limit: Some(100),
            ..Default::default()
        }).await?;

        // Group orders by destination region
        let mut by_region: HashMap<String, Vec<&Order>> = HashMap::new();
        for order in &orders.items {
            let region = order.shipping_address.as_ref()
                .map(|a| a.state.clone().unwrap_or_else(|| a.country.clone()))
                .unwrap_or_else(|| "Unknown".to_string());
            by_region.entry(region).or_default().push(order);
        }

        // Identify batching opportunities
        for (region, regional_orders) in by_region {
            if regional_orders.len() >= 5 {
                self.base.emit_alert(
                    "info",
                    &format!(
                        "Batch opportunity: {} orders for region {} could be consolidated",
                        regional_orders.len(), region
                    ),
                ).await;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Agent for OrderRoutingAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        "Routes orders to optimal fulfillment locations"
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
        // 1. Route pending orders
        let decisions = self.route_pending_orders().await?;
        info!("Generated {} routing decisions", decisions.len());

        // 2. Execute routing
        if !decisions.is_empty() {
            self.execute_routing(decisions).await?;
        }

        // 3. Handle unfulfillable orders
        if let Err(e) = self.handle_unfulfillable_orders().await {
            warn!("Unfulfillable order handling failed: {}", e);
        }

        // 4. Optimize batch routing
        if let Err(e) = self.optimize_batch_routing().await {
            warn!("Batch optimization failed: {}", e);
        }

        Ok(vec![])
    }

    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()> {
        info!("Executing routing decision: {}", decision.action);
        Ok(())
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct RoutingAnalysis {
    strategy: String,
    primary_location: Option<String>,
    allocations: Vec<InventoryAllocation>,
    #[allow(dead_code)]
    split_reason: Option<String>,
    #[allow(dead_code)]
    estimated_ship_date: Option<String>,
    priority: String,
    confidence: f32,
    reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InventoryAllocation {
    product_id: String,
    location_id: String,
    quantity: i32,
}

#[derive(Debug)]
struct RoutingDecision {
    order_id: String,
    strategy: String,
    primary_location: Option<String>,
    allocations: Vec<InventoryAllocation>,
    priority: String,
    confidence: f32,
    reasoning: String,
}
