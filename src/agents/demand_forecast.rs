//! Demand Forecasting Agent
//!
//! AI-powered demand prediction:
//! - Sales trend analysis
//! - Seasonal pattern detection
//! - Inventory optimization recommendations
//! - Promotional impact prediction

use super::base::{Agent, AgentContext, AgentDecision, AgentEvent, BaseAgent};
use crate::clients::{
    CommerceApi, DashboardMetrics, ListProductsParams, Product, SalesTrend, StateSetClient,
};
use crate::config::AgentInstanceConfig;
use crate::services::LLMService;
use async_trait::async_trait;
use chrono::{Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Demand forecasting agent
pub struct DemandForecastAgent {
    base: BaseAgent,
}

impl DemandForecastAgent {
    pub fn new(
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            base: BaseAgent::new("DemandForecastAgent", client, llm, event_tx, config),
        }
    }

    /// Generate demand forecasts for products
    async fn generate_forecasts(&self) -> anyhow::Result<Vec<DemandForecast>> {
        // Get historical sales data
        let end_date = Utc::now();
        let start_date = end_date - ChronoDuration::days(90);

        let trends = self.base.client.get_sales_trends(
            &start_date.format("%Y-%m-%d").to_string(),
            &end_date.format("%Y-%m-%d").to_string(),
            "week",
        ).await?;

        // Get current metrics
        let metrics = self.base.client.get_dashboard_metrics().await?;

        // Get products
        let products = self.base.client.list_products(ListProductsParams {
            active: Some(true),
            limit: Some(100),
            ..Default::default()
        }).await?;

        let mut forecasts = Vec::new();

        // Generate forecast using AI
        let context = self.build_forecast_context(&trends, &metrics, &products.items);

        let system_prompt = r#"You are a Demand Forecasting AI for an e-commerce platform. Analyze the historical data and generate demand predictions.

Consider:
1. Historical sales trends and patterns
2. Seasonal variations
3. Current inventory levels
4. Market conditions
5. Growth trajectory

For each significant product category, provide:
{
    "forecasts": [
        {
            "category": "category name",
            "current_weekly_demand": 100,
            "predicted_weekly_demand": 120,
            "trend": "increasing/stable/decreasing",
            "seasonality": "high/medium/low",
            "confidence": 0.0-1.0,
            "recommendation": "action to take"
        }
    ],
    "overall_trend": "growth/stable/decline",
    "key_insights": ["insight 1", "insight 2"],
    "risk_factors": ["risk 1", "risk 2"]
}"#;

        let response = self.base.llm.chat_completion(system_prompt, &context).await?;

        if let Ok(analysis) = self.base.parse_llm_json::<ForecastAnalysis>(&response) {
            for forecast in analysis.forecasts {
                let decision = AgentDecision::new(
                    "demand_forecast",
                    &forecast.recommendation,
                    forecast.confidence,
                    &format!(
                        "Category {}: {} trend, demand {} -> {}",
                        forecast.category, forecast.trend,
                        forecast.current_weekly_demand, forecast.predicted_weekly_demand
                    ),
                );

                self.base.emit_decision(&decision).await;

                forecasts.push(DemandForecast {
                    category: forecast.category,
                    current_demand: forecast.current_weekly_demand,
                    predicted_demand: forecast.predicted_weekly_demand,
                    trend: forecast.trend,
                    seasonality: forecast.seasonality,
                    confidence: forecast.confidence,
                    recommendation: forecast.recommendation,
                });
            }

            // Emit key insights as alerts
            for insight in analysis.key_insights {
                self.base.emit_alert("info", &insight).await;
            }

            // Emit risk factors as warnings
            for risk in analysis.risk_factors {
                self.base.emit_alert("warning", &format!("Risk factor: {}", risk)).await;
            }
        }

        Ok(forecasts)
    }

    /// Build context for forecast analysis
    fn build_forecast_context(
        &self,
        trends: &[SalesTrend],
        metrics: &DashboardMetrics,
        products: &[Product],
    ) -> String {
        let trends_summary: Vec<String> = trends.iter().map(|t| {
            format!("{}: {} orders, ${:.2} revenue", t.period, t.order_count, t.revenue)
        }).collect();

        let categories: std::collections::HashSet<String> = products.iter()
            .filter_map(|p| p.category.clone())
            .collect();

        format!(
            r#"Current Metrics:
- Total Orders: {}
- Total Revenue: ${:.2}
- Average Order Value: ${:.2}
- Orders Today: {}
- Low Stock Items: {}

Weekly Sales Trends (Last 90 Days):
{}

Active Product Categories: {:?}
Total Active Products: {}"#,
            metrics.total_orders,
            metrics.total_revenue,
            metrics.average_order_value,
            metrics.orders_today,
            metrics.low_stock_items,
            trends_summary.join("\n"),
            categories,
            products.len()
        )
    }

    /// Detect anomalies in sales patterns
    async fn detect_anomalies(&self) -> anyhow::Result<Vec<Anomaly>> {
        let end_date = Utc::now();
        let start_date = end_date - ChronoDuration::days(30);

        let trends = self.base.client.get_sales_trends(
            &start_date.format("%Y-%m-%d").to_string(),
            &end_date.format("%Y-%m-%d").to_string(),
            "day",
        ).await?;

        let mut anomalies = Vec::new();

        // Calculate statistics
        if trends.len() < 7 {
            return Ok(anomalies);
        }

        let revenues: Vec<f64> = trends.iter().map(|t| t.revenue).collect();
        let mean = revenues.iter().sum::<f64>() / revenues.len() as f64;
        let variance = revenues.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / revenues.len() as f64;
        let std_dev = variance.sqrt();

        // Detect outliers (> 2 std devs from mean)
        for trend in &trends {
            let z_score = (trend.revenue - mean) / std_dev.max(1.0);

            if z_score.abs() > 2.0 {
                let anomaly = Anomaly {
                    period: trend.period.clone(),
                    metric: "revenue".to_string(),
                    actual_value: trend.revenue,
                    expected_value: mean,
                    deviation: z_score,
                    severity: if z_score.abs() > 3.0 { "high" } else { "medium" }.to_string(),
                };

                self.base.emit_alert(
                    &anomaly.severity,
                    &format!(
                        "Anomaly detected on {}: ${:.2} (expected ${:.2}, {:.1}σ deviation)",
                        anomaly.period, anomaly.actual_value, anomaly.expected_value, z_score
                    ),
                ).await;

                anomalies.push(anomaly);
            }
        }

        self.base.emit_metric("anomalies_detected", anomalies.len() as f64).await;

        Ok(anomalies)
    }

    /// Generate inventory optimization recommendations
    async fn optimize_inventory(&self) -> anyhow::Result<()> {
        let forecasts = self.generate_forecasts().await?;

        for forecast in forecasts {
            let growth_rate = if forecast.current_demand > 0 {
                ((forecast.predicted_demand as f64 - forecast.current_demand as f64)
                    / forecast.current_demand as f64) * 100.0
            } else {
                0.0
            };

            // Generate recommendations based on trend
            let recommendation = match forecast.trend.as_str() {
                "increasing" if growth_rate > 20.0 => {
                    format!(
                        "INCREASE_STOCK: Category {} showing {:.1}% growth. Recommend increasing safety stock.",
                        forecast.category, growth_rate
                    )
                }
                "decreasing" if growth_rate < -20.0 => {
                    format!(
                        "REDUCE_ORDERS: Category {} declining {:.1}%. Consider reducing replenishment quantities.",
                        forecast.category, growth_rate.abs()
                    )
                }
                _ => {
                    format!(
                        "MAINTAIN: Category {} is stable. Continue current inventory strategy.",
                        forecast.category
                    )
                }
            };

            let decision = AgentDecision::new(
                "inventory_optimization",
                &recommendation,
                forecast.confidence,
                &format!("Based on {} trend with {:.1}% change", forecast.trend, growth_rate),
            );

            self.base.emit_decision(&decision).await;
        }

        Ok(())
    }
}

#[async_trait]
impl Agent for DemandForecastAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        "Predicts demand patterns and optimizes inventory planning"
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

        // 1. Generate demand forecasts
        let forecasts = self.generate_forecasts().await?;
        info!("Generated {} demand forecasts", forecasts.len());

        // 2. Detect anomalies
        let anomalies = self.detect_anomalies().await?;
        if !anomalies.is_empty() {
            info!("Detected {} anomalies", anomalies.len());
        }

        // 3. Generate inventory optimization recommendations
        if let Err(e) = self.optimize_inventory().await {
            warn!("Inventory optimization failed: {}", e);
        }

        Ok(decisions)
    }

    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()> {
        info!("Executing forecast decision: {}", decision.action);
        Ok(())
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct ForecastAnalysis {
    forecasts: Vec<CategoryForecast>,
    #[allow(dead_code)]
    overall_trend: String,
    key_insights: Vec<String>,
    risk_factors: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CategoryForecast {
    category: String,
    current_weekly_demand: i32,
    predicted_weekly_demand: i32,
    trend: String,
    seasonality: String,
    confidence: f32,
    recommendation: String,
}

#[derive(Debug)]
struct DemandForecast {
    category: String,
    current_demand: i32,
    predicted_demand: i32,
    trend: String,
    seasonality: String,
    confidence: f32,
    recommendation: String,
}

#[derive(Debug)]
struct Anomaly {
    period: String,
    metric: String,
    actual_value: f64,
    expected_value: f64,
    deviation: f64,
    severity: String,
}
