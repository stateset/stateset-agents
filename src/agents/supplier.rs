//! Supplier Management Agent
//!
//! AI agent for supplier relationship management:
//! - Supplier performance monitoring
//! - Automated purchase order creation
//! - Lead time optimization
//! - Risk assessment

use super::base::{Agent, AgentContext, AgentDecision, AgentEvent, BaseAgent};
use crate::clients::{
    CommerceApi, CreatePurchaseOrderRequest, InventoryItem, ListInventoryParams,
    ListPurchaseOrdersParams, PurchaseOrder, PurchaseOrderItemRequest, PurchaseOrderStatus,
    StateSetClient,
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

/// Supplier management agent
pub struct SupplierManagementAgent {
    base: BaseAgent,
}

impl SupplierManagementAgent {
    pub fn new(
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            base: BaseAgent::new("SupplierManagementAgent", client, llm, event_tx, config),
        }
    }

    /// Analyze inventory and create replenishment orders
    async fn analyze_replenishment_needs(&self) -> anyhow::Result<Vec<ReplenishmentRecommendation>> {
        let low_stock = self.base.client.get_low_stock().await?;

        if low_stock.is_empty() {
            info!("No low stock items found");
            return Ok(vec![]);
        }

        let mut recommendations = Vec::new();

        // Group by supplier (in production, would have supplier mapping)
        let mut by_supplier: HashMap<String, Vec<&InventoryItem>> = HashMap::new();
        for item in &low_stock {
            // Placeholder - would look up preferred supplier for product
            let supplier_id = format!("supplier_{}", &item.product_id[..8.min(item.product_id.len())]);
            by_supplier.entry(supplier_id).or_default().push(item);
        }

        for (supplier_id, items) in by_supplier {
            let context = self.build_replenishment_context(&supplier_id, &items);

            let system_prompt = r#"You are a Supply Chain AI Agent. Analyze the inventory situation and recommend a purchase order.

Consider:
1. Current stock levels vs reorder points
2. Historical demand patterns
3. Supplier lead times
4. Economic order quantities
5. Safety stock requirements

Respond with JSON:
{
    "should_order": true/false,
    "items": [
        {"product_id": "...", "quantity": 100, "urgency": "high/medium/low"}
    ],
    "total_quantity": 0,
    "estimated_value": 0.00,
    "reasoning": "explanation",
    "confidence": 0.0-1.0
}"#;

            let response = self.base.llm.chat_completion(system_prompt, &context).await?;

            if let Ok(analysis) = self.base.parse_llm_json::<ReplenishmentAnalysis>(&response) {
                if analysis.should_order && analysis.confidence >= self.base.config.confidence_threshold {
                    recommendations.push(ReplenishmentRecommendation {
                        supplier_id: supplier_id.clone(),
                        items: analysis.items,
                        total_quantity: analysis.total_quantity,
                        estimated_value: analysis.estimated_value,
                        reasoning: analysis.reasoning,
                        confidence: analysis.confidence,
                    });
                }
            }
        }

        Ok(recommendations)
    }

    /// Build context for replenishment analysis
    fn build_replenishment_context(&self, supplier_id: &str, items: &[&InventoryItem]) -> String {
        let items_info: Vec<String> = items.iter().map(|item| {
            format!(
                "- SKU: {}, On Hand: {}, Reserved: {}, Available: {}, Reorder Point: {}",
                item.sku,
                item.quantity_on_hand,
                item.quantity_reserved,
                item.quantity_available,
                item.reorder_point.unwrap_or(10)
            )
        }).collect();

        format!(
            "Supplier: {}\n\nLow Stock Items:\n{}",
            supplier_id,
            items_info.join("\n")
        )
    }

    /// Create purchase orders for approved recommendations
    async fn create_purchase_orders(&self, recommendations: Vec<ReplenishmentRecommendation>) -> anyhow::Result<Vec<PurchaseOrder>> {
        let mut created_pos = Vec::new();

        for rec in recommendations {
            let decision = AgentDecision::new(
                "purchase_order",
                "create_po",
                rec.confidence,
                &rec.reasoning,
            ).with_metadata(serde_json::json!({
                "supplier_id": rec.supplier_id,
                "total_quantity": rec.total_quantity,
                "estimated_value": rec.estimated_value,
            })).with_approval(rec.estimated_value > 10000.0);

            self.base.emit_decision(&decision).await;

            // Only auto-create for lower value POs
            if rec.estimated_value <= 10000.0 && !self.base.requires_approval(&decision) {
                let po = self.base.client.create_purchase_order(CreatePurchaseOrderRequest {
                    supplier_id: rec.supplier_id.clone(),
                    items: rec.items.iter().map(|i| PurchaseOrderItemRequest {
                        product_id: i.product_id.clone(),
                        quantity: i.quantity,
                        unit_cost: 10.0, // Placeholder - would lookup actual cost
                    }).collect(),
                    notes: Some(format!("Auto-generated by SupplierAgent: {}", rec.reasoning)),
                    expected_delivery: None,
                }).await?;

                info!("Created PO {} for supplier {}", po.id, rec.supplier_id);
                self.base.emit_action("create_po", Ok(po.id.clone())).await;
                created_pos.push(po);
            } else {
                info!(
                    "PO for supplier {} requires approval (value: ${:.2})",
                    rec.supplier_id, rec.estimated_value
                );
            }
        }

        self.base.emit_metric("pos_created", created_pos.len() as f64).await;
        Ok(created_pos)
    }

    /// Monitor supplier performance
    async fn monitor_supplier_performance(&self) -> anyhow::Result<()> {
        let pos = self.base.client.list_purchase_orders(ListPurchaseOrdersParams {
            limit: Some(100),
            ..Default::default()
        }).await?;

        // Group by supplier and analyze
        let mut by_supplier: HashMap<String, Vec<&PurchaseOrder>> = HashMap::new();
        for po in &pos.items {
            by_supplier.entry(po.supplier_id.clone()).or_default().push(po);
        }

        for (supplier_id, supplier_pos) in by_supplier {
            let total = supplier_pos.len();
            let received = supplier_pos.iter()
                .filter(|po| po.status == PurchaseOrderStatus::Received)
                .count();
            let canceled = supplier_pos.iter()
                .filter(|po| po.status == PurchaseOrderStatus::Canceled)
                .count();

            let fulfillment_rate = if total > 0 {
                (received as f64 / total as f64) * 100.0
            } else {
                100.0
            };

            let cancellation_rate = if total > 0 {
                (canceled as f64 / total as f64) * 100.0
            } else {
                0.0
            };

            // Alert on poor performance
            if fulfillment_rate < 80.0 {
                self.base.emit_alert(
                    "warning",
                    &format!(
                        "Supplier {} has low fulfillment rate: {:.1}%",
                        supplier_id, fulfillment_rate
                    ),
                ).await;
            }

            if cancellation_rate > 10.0 {
                self.base.emit_alert(
                    "warning",
                    &format!(
                        "Supplier {} has high cancellation rate: {:.1}%",
                        supplier_id, cancellation_rate
                    ),
                ).await;
            }

            self.base.emit_metric(
                &format!("supplier_{}_fulfillment_rate", supplier_id),
                fulfillment_rate,
            ).await;
        }

        Ok(())
    }

    /// Assess supplier risk
    async fn assess_supplier_risk(&self) -> anyhow::Result<()> {
        let pos = self.base.client.list_purchase_orders(ListPurchaseOrdersParams {
            status: Some(PurchaseOrderStatus::Confirmed),
            limit: Some(50),
            ..Default::default()
        }).await?;

        for po in pos.items {
            // Check for overdue deliveries
            if let Some(expected) = &po.expected_delivery {
                // In production, would parse and compare dates
                let is_overdue = false; // Placeholder

                if is_overdue {
                    let system_prompt = r#"You are a Supply Chain Risk Analyst. Assess the risk of this overdue purchase order.

Provide:
{
    "risk_level": "low/medium/high/critical",
    "impact": "description of business impact",
    "mitigation": "recommended actions",
    "confidence": 0.0-1.0
}"#;

                    let po_json = serde_json::to_string_pretty(&po)?;
                    let response = self.base.llm.chat_completion(system_prompt, &po_json).await?;

                    #[derive(Deserialize)]
                    struct RiskAssessment {
                        risk_level: String,
                        impact: String,
                        mitigation: String,
                        confidence: f32,
                    }

                    if let Ok(assessment) = self.base.parse_llm_json::<RiskAssessment>(&response) {
                        if assessment.risk_level == "high" || assessment.risk_level == "critical" {
                            self.base.emit_alert(
                                &assessment.risk_level,
                                &format!("PO {} risk: {} - Mitigation: {}",
                                    po.id, assessment.impact, assessment.mitigation
                                ),
                            ).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Agent for SupplierManagementAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        "Manages supplier relationships and automates procurement"
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

        // 1. Analyze replenishment needs
        let recommendations = self.analyze_replenishment_needs().await?;
        info!("Generated {} replenishment recommendations", recommendations.len());

        // 2. Create purchase orders
        let _created = self.create_purchase_orders(recommendations).await?;

        // 3. Monitor supplier performance
        if let Err(e) = self.monitor_supplier_performance().await {
            warn!("Supplier monitoring failed: {}", e);
        }

        // 4. Assess supplier risk
        if let Err(e) = self.assess_supplier_risk().await {
            warn!("Risk assessment failed: {}", e);
        }

        Ok(decisions)
    }

    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()> {
        info!("Executing supplier decision: {}", decision.action);
        Ok(())
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct ReplenishmentAnalysis {
    should_order: bool,
    items: Vec<ReplenishmentItem>,
    total_quantity: i32,
    estimated_value: f64,
    reasoning: String,
    confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplenishmentItem {
    product_id: String,
    quantity: i32,
    urgency: String,
}

#[derive(Debug)]
struct ReplenishmentRecommendation {
    supplier_id: String,
    items: Vec<ReplenishmentItem>,
    total_quantity: i32,
    estimated_value: f64,
    reasoning: String,
    confidence: f32,
}
