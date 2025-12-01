//! Quality Control Agent
//!
//! AI agent for quality assurance:
//! - Return pattern analysis
//! - Product quality monitoring
//! - Defect prediction
//! - Supplier quality scoring

use super::base::{Agent, AgentContext, AgentDecision, AgentEvent, BaseAgent};
use crate::clients::{
    CommerceApi, ListReturnsParams, Product, Return, ReturnStatus, StateSetClient,
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

/// Quality control agent
pub struct QualityControlAgent {
    base: BaseAgent,
}

impl QualityControlAgent {
    pub fn new(
        client: Arc<StateSetClient>,
        llm: Arc<LLMService>,
        event_tx: mpsc::Sender<AgentEvent>,
        config: AgentInstanceConfig,
    ) -> Self {
        Self {
            base: BaseAgent::new("QualityControlAgent", client, llm, event_tx, config),
        }
    }

    /// Analyze return patterns to identify quality issues
    async fn analyze_return_patterns(&self) -> anyhow::Result<Vec<QualityIssue>> {
        let returns = self.base.client.list_returns(ListReturnsParams {
            limit: Some(500),
            ..Default::default()
        }).await?;

        if returns.items.is_empty() {
            return Ok(vec![]);
        }

        // Group returns by product/reason
        let mut by_reason: HashMap<String, Vec<&Return>> = HashMap::new();
        let mut by_product: HashMap<String, Vec<&Return>> = HashMap::new();

        for ret in &returns.items {
            by_reason.entry(ret.reason.clone()).or_default().push(ret);

            for item in &ret.items {
                by_product.entry(item.order_item_id.clone()).or_default().push(ret);
            }
        }

        let mut issues = Vec::new();

        // Analyze reason patterns
        let total_returns = returns.items.len();
        for (reason, returns_for_reason) in &by_reason {
            let percentage = (returns_for_reason.len() as f64 / total_returns as f64) * 100.0;

            if percentage > 15.0 {
                // Use AI to analyze if this is a quality issue
                let context = format!(
                    "Return Reason: {}\nCount: {} ({:.1}% of total returns)\nSample returns:\n{}",
                    reason,
                    returns_for_reason.len(),
                    percentage,
                    returns_for_reason.iter().take(5)
                        .map(|r| format!("- Order {}: {}", r.order_id, r.customer_notes.as_deref().unwrap_or("")))
                        .collect::<Vec<_>>()
                        .join("\n")
                );

                let system_prompt = r#"You are a Quality Control AI Analyst. Analyze this return pattern and determine if it indicates a product quality issue.

Categories of issues:
- DEFECT: Manufacturing defect
- DAMAGE: Shipping/handling damage
- MISMATCH: Product doesn't match description
- PERFORMANCE: Product underperforms
- PREFERENCE: Customer preference (not quality related)

Respond with JSON:
{
    "is_quality_issue": true/false,
    "category": "DEFECT/DAMAGE/MISMATCH/PERFORMANCE/PREFERENCE",
    "severity": "low/medium/high/critical",
    "root_cause": "identified root cause",
    "recommended_action": "what to do",
    "confidence": 0.0-1.0
}"#;

                let response = self.base.llm.chat_completion(system_prompt, &context).await?;

                if let Ok(analysis) = self.base.parse_llm_json::<QualityAnalysis>(&response) {
                    if analysis.is_quality_issue {
                        issues.push(QualityIssue {
                            category: analysis.category,
                            severity: analysis.severity,
                            affected_returns: returns_for_reason.len(),
                            percentage,
                            root_cause: analysis.root_cause,
                            recommended_action: analysis.recommended_action,
                            confidence: analysis.confidence,
                        });
                    }
                }
            }
        }

        Ok(issues)
    }

    /// Score products by quality based on return data
    async fn score_product_quality(&self) -> anyhow::Result<Vec<ProductQualityScore>> {
        let returns = self.base.client.list_returns(ListReturnsParams {
            limit: Some(500),
            ..Default::default()
        }).await?;

        // In production, would cross-reference with order data to get return rate
        let mut product_returns: HashMap<String, i32> = HashMap::new();
        for ret in &returns.items {
            for item in &ret.items {
                *product_returns.entry(item.order_item_id.clone()).or_insert(0) += item.quantity;
            }
        }

        let mut scores = Vec::new();

        for (product_id, return_count) in product_returns {
            // Simplified scoring - in production would use total sales
            let return_rate = return_count as f64 * 2.0; // Placeholder percentage
            let quality_score = (100.0 - return_rate.min(100.0)).max(0.0);

            let grade = match quality_score {
                s if s >= 95.0 => "A",
                s if s >= 85.0 => "B",
                s if s >= 70.0 => "C",
                s if s >= 50.0 => "D",
                _ => "F",
            };

            if quality_score < 70.0 {
                self.base.emit_alert(
                    "warning",
                    &format!("Product {} has quality score {:.1} (grade {})", product_id, quality_score, grade),
                ).await;
            }

            scores.push(ProductQualityScore {
                product_id,
                return_count,
                quality_score,
                grade: grade.to_string(),
            });
        }

        Ok(scores)
    }

    /// Monitor supplier quality
    async fn monitor_supplier_quality(&self) -> anyhow::Result<()> {
        let returns = self.base.client.list_returns(ListReturnsParams {
            status: Some(ReturnStatus::Completed),
            limit: Some(200),
            ..Default::default()
        }).await?;

        // Group by reason category
        let defect_returns: Vec<_> = returns.items.iter()
            .filter(|r| {
                let reason_lower = r.reason.to_lowercase();
                reason_lower.contains("defect") ||
                reason_lower.contains("broken") ||
                reason_lower.contains("damaged") ||
                reason_lower.contains("not working")
            })
            .collect();

        let defect_rate = if !returns.items.is_empty() {
            (defect_returns.len() as f64 / returns.items.len() as f64) * 100.0
        } else {
            0.0
        };

        self.base.emit_metric("defect_return_rate", defect_rate).await;

        if defect_rate > 5.0 {
            self.base.emit_alert(
                "warning",
                &format!("High defect return rate: {:.1}%", defect_rate),
            ).await;

            // Analyze patterns for root cause
            let context = format!(
                "Defect returns ({:.1}% of total):\n{}",
                defect_rate,
                defect_returns.iter().take(10)
                    .map(|r| format!("- {}: {}", r.reason, r.customer_notes.as_deref().unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join("\n")
            );

            let system_prompt = r#"You are a Quality Control expert. Analyze these defect returns and identify:
1. Common patterns
2. Likely root causes
3. Recommended corrective actions

Respond with JSON:
{
    "patterns": ["pattern 1", "pattern 2"],
    "root_causes": ["cause 1", "cause 2"],
    "corrective_actions": ["action 1", "action 2"],
    "priority": "immediate/high/medium/low"
}"#;

            let response = self.base.llm.chat_completion(system_prompt, &context).await?;

            #[derive(Deserialize)]
            struct DefectAnalysis {
                patterns: Vec<String>,
                root_causes: Vec<String>,
                corrective_actions: Vec<String>,
                priority: String,
            }

            if let Ok(analysis) = self.base.parse_llm_json::<DefectAnalysis>(&response) {
                let decision = AgentDecision::new(
                    "quality_improvement",
                    &analysis.corrective_actions.join("; "),
                    0.85,
                    &format!(
                        "Root causes: {}. Priority: {}",
                        analysis.root_causes.join(", "),
                        analysis.priority
                    ),
                ).with_approval(analysis.priority == "immediate");

                self.base.emit_decision(&decision).await;
            }
        }

        Ok(())
    }

    /// Predict potential quality issues
    async fn predict_quality_issues(&self) -> anyhow::Result<()> {
        // Would use historical data to predict future issues
        // For now, emit metrics on current quality state
        let scores = self.score_product_quality().await?;

        let avg_score = if !scores.is_empty() {
            scores.iter().map(|s| s.quality_score).sum::<f64>() / scores.len() as f64
        } else {
            100.0
        };

        self.base.emit_metric("average_quality_score", avg_score).await;

        let low_quality_count = scores.iter().filter(|s| s.quality_score < 70.0).count();
        self.base.emit_metric("low_quality_products", low_quality_count as f64).await;

        Ok(())
    }
}

#[async_trait]
impl Agent for QualityControlAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        "Monitors and improves product quality through return analysis"
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

        // 1. Analyze return patterns
        let issues = self.analyze_return_patterns().await?;
        info!("Found {} quality issues", issues.len());

        for issue in &issues {
            let decision = AgentDecision::new(
                "quality_issue",
                &issue.recommended_action,
                issue.confidence,
                &format!(
                    "{} issue ({}): {}. Affects {} returns ({:.1}%)",
                    issue.category, issue.severity, issue.root_cause,
                    issue.affected_returns, issue.percentage
                ),
            ).with_approval(issue.severity == "critical" || issue.severity == "high");

            self.base.emit_decision(&decision).await;
            decisions.push(decision);
        }

        // 2. Monitor supplier quality
        if let Err(e) = self.monitor_supplier_quality().await {
            warn!("Supplier quality monitoring failed: {}", e);
        }

        // 3. Predict quality issues
        if let Err(e) = self.predict_quality_issues().await {
            warn!("Quality prediction failed: {}", e);
        }

        Ok(decisions)
    }

    async fn execute_decision(&self, decision: &AgentDecision) -> anyhow::Result<()> {
        info!("Executing quality decision: {}", decision.action);
        Ok(())
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct QualityAnalysis {
    is_quality_issue: bool,
    category: String,
    severity: String,
    root_cause: String,
    recommended_action: String,
    confidence: f32,
}

#[derive(Debug)]
struct QualityIssue {
    category: String,
    severity: String,
    affected_returns: usize,
    percentage: f64,
    root_cause: String,
    recommended_action: String,
    confidence: f32,
}

#[derive(Debug)]
struct ProductQualityScore {
    product_id: String,
    return_count: i32,
    quality_score: f64,
    grade: String,
}
