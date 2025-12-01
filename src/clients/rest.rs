//! REST Client for StateSet API
//!
//! Full-featured HTTP client with retry logic, rate limiting, and idempotency support.

use super::*;
use crate::config::ApiConfig;
use async_trait::async_trait;
use reqwest::{Client, Response, StatusCode};
use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// REST client for StateSet API
#[derive(Clone)]
pub struct RestClient {
    client: Client,
    base_url: String,
    api_key: String,
    max_retries: u32,
}

impl RestClient {
    /// Create a new REST client from configuration
    pub fn new(config: &ApiConfig) -> anyhow::Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(10)
            .build()?;

        Ok(Self {
            client,
            base_url: config.rest_url.trim_end_matches('/').to_string(),
            api_key: config.api_key.clone(),
            max_retries: config.max_retries,
        })
    }

    /// Create a new REST client with custom parameters
    pub fn with_params(base_url: &str, api_key: &str, timeout_secs: u64) -> anyhow::Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            max_retries: 3,
        })
    }

    /// Make a GET request
    async fn get<T: DeserializeOwned>(&self, path: &str) -> anyhow::Result<T> {
        let url = format!("{}{}", self.base_url, path);
        debug!("GET {}", url);

        let response = self.execute_with_retry(|| async {
            self.client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("X-API-Key", &self.api_key)
                .send()
                .await
        }).await?;

        self.handle_response(response).await
    }

    /// Make a POST request
    async fn post<T: DeserializeOwned, B: serde::Serialize + Send + Sync>(
        &self,
        path: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let idempotency_key = Uuid::new_v4().to_string();
        debug!("POST {} (idempotency: {})", url, idempotency_key);

        let response = self.execute_with_retry(|| async {
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("X-API-Key", &self.api_key)
                .header("Idempotency-Key", &idempotency_key)
                .header("Content-Type", "application/json")
                .json(body)
                .send()
                .await
        }).await?;

        self.handle_response(response).await
    }

    /// Make a PUT request
    async fn put<T: DeserializeOwned, B: serde::Serialize + Send + Sync>(
        &self,
        path: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        let url = format!("{}{}", self.base_url, path);
        debug!("PUT {}", url);

        let response = self.execute_with_retry(|| async {
            self.client
                .put(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("X-API-Key", &self.api_key)
                .header("Content-Type", "application/json")
                .json(body)
                .send()
                .await
        }).await?;

        self.handle_response(response).await
    }

    /// Make a DELETE request
    async fn delete(&self, path: &str) -> anyhow::Result<()> {
        let url = format!("{}{}", self.base_url, path);
        debug!("DELETE {}", url);

        let response = self.execute_with_retry(|| async {
            self.client
                .delete(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("X-API-Key", &self.api_key)
                .send()
                .await
        }).await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            Err(anyhow::anyhow!("DELETE failed with status {}: {}", status, text))
        }
    }

    /// Execute request with exponential backoff retry
    async fn execute_with_retry<F, Fut>(&self, request_fn: F) -> anyhow::Result<Response>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<Response, reqwest::Error>>,
    {
        let mut last_error = None;
        let mut delay = Duration::from_millis(100);

        for attempt in 0..=self.max_retries {
            match request_fn().await {
                Ok(response) => {
                    let status = response.status();

                    // Don't retry client errors (except rate limiting)
                    if status.is_client_error() && status != StatusCode::TOO_MANY_REQUESTS {
                        return Ok(response);
                    }

                    // Success or non-retryable
                    if status.is_success() || attempt == self.max_retries {
                        return Ok(response);
                    }

                    // Rate limited - respect Retry-After header
                    if status == StatusCode::TOO_MANY_REQUESTS {
                        if let Some(retry_after) = response.headers().get("Retry-After") {
                            if let Ok(secs) = retry_after.to_str().unwrap_or("1").parse::<u64>() {
                                delay = Duration::from_secs(secs);
                            }
                        }
                        warn!("Rate limited, retrying after {:?}", delay);
                    }

                    // Server error - retry with backoff
                    if status.is_server_error() {
                        warn!("Server error {}, attempt {}/{}", status, attempt + 1, self.max_retries);
                    }
                }
                Err(e) => {
                    warn!("Request failed: {}, attempt {}/{}", e, attempt + 1, self.max_retries);
                    last_error = Some(e);
                }
            }

            if attempt < self.max_retries {
                tokio::time::sleep(delay).await;
                delay = std::cmp::min(delay * 2, Duration::from_secs(30));
            }
        }

        Err(anyhow::anyhow!(
            "Request failed after {} retries: {:?}",
            self.max_retries,
            last_error
        ))
    }

    /// Handle response and deserialize
    async fn handle_response<T: DeserializeOwned>(&self, response: Response) -> anyhow::Result<T> {
        let status = response.status();
        let url = response.url().to_string();

        if status.is_success() {
            let body = response.text().await?;
            debug!("Response body: {}", &body[..std::cmp::min(500, body.len())]);

            // Try to parse as wrapped response first
            #[derive(Deserialize)]
            struct WrappedResponse<T> {
                success: Option<bool>,
                data: Option<T>,
                #[serde(flatten)]
                direct: Option<T>,
            }

            // First try direct deserialization
            if let Ok(data) = serde_json::from_str::<T>(&body) {
                return Ok(data);
            }

            // Then try wrapped format
            let wrapped: WrappedResponse<T> = serde_json::from_str(&body)
                .map_err(|e| anyhow::anyhow!("Failed to parse response: {} - Body: {}", e, body))?;

            wrapped.data.or(wrapped.direct)
                .ok_or_else(|| anyhow::anyhow!("Response missing data field"))
        } else {
            let body = response.text().await.unwrap_or_default();
            error!("API error {} at {}: {}", status, url, body);
            Err(anyhow::anyhow!("API error {}: {}", status, body))
        }
    }

    /// Build query string from params
    fn build_query<P: serde::Serialize>(&self, params: &P) -> String {
        serde_urlencoded::to_string(params).unwrap_or_default()
    }
}

#[async_trait]
impl CommerceApi for RestClient {
    // ========================================================================
    // Orders
    // ========================================================================

    async fn create_order(&self, request: CreateOrderRequest) -> anyhow::Result<Order> {
        self.post("/orders", &request).await
    }

    async fn get_order(&self, order_id: &str) -> anyhow::Result<Order> {
        self.get(&format!("/orders/{}", order_id)).await
    }

    async fn list_orders(&self, params: ListOrdersParams) -> anyhow::Result<OrderList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/orders".to_string()
        } else {
            format!("/orders?{}", query)
        };
        self.get(&path).await
    }

    async fn update_order_status(
        &self,
        order_id: &str,
        status: OrderStatus,
        notes: Option<String>,
    ) -> anyhow::Result<Order> {
        #[derive(Serialize)]
        struct UpdateRequest {
            status: OrderStatus,
            #[serde(skip_serializing_if = "Option::is_none")]
            notes: Option<String>,
        }

        self.put(
            &format!("/orders/{}/status", order_id),
            &UpdateRequest { status, notes },
        ).await
    }

    async fn cancel_order(&self, order_id: &str, reason: &str) -> anyhow::Result<Order> {
        #[derive(Serialize)]
        struct CancelRequest {
            reason: String,
            refund: bool,
        }

        self.post(
            &format!("/orders/{}/cancel", order_id),
            &CancelRequest {
                reason: reason.to_string(),
                refund: true,
            },
        ).await
    }

    // ========================================================================
    // Inventory
    // ========================================================================

    async fn get_inventory(&self, product_id: &str, location_id: Option<&str>) -> anyhow::Result<InventoryItem> {
        let path = match location_id {
            Some(loc) => format!("/inventory?product_id={}&location_id={}", product_id, loc),
            None => format!("/inventory?product_id={}", product_id),
        };
        let list: InventoryList = self.get(&path).await?;
        list.items.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("Inventory not found for product {}", product_id))
    }

    async fn list_inventory(&self, params: ListInventoryParams) -> anyhow::Result<InventoryList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/inventory".to_string()
        } else {
            format!("/inventory?{}", query)
        };
        self.get(&path).await
    }

    async fn reserve_inventory(&self, request: ReserveInventoryRequest) -> anyhow::Result<InventoryReservation> {
        self.post(
            &format!("/inventory/{}/reserve", request.inventory_id),
            &request,
        ).await
    }

    async fn release_inventory(&self, reservation_id: &str) -> anyhow::Result<()> {
        self.post(
            &format!("/inventory/reservations/{}/release", reservation_id),
            &serde_json::json!({}),
        ).await
    }

    async fn adjust_inventory(&self, request: AdjustInventoryRequest) -> anyhow::Result<InventoryItem> {
        self.post("/inventory/adjust", &request).await
    }

    async fn get_low_stock(&self) -> anyhow::Result<Vec<InventoryItem>> {
        let list: InventoryList = self.get("/inventory/low-stock").await?;
        Ok(list.items)
    }

    // ========================================================================
    // Returns
    // ========================================================================

    async fn create_return(&self, request: CreateReturnRequest) -> anyhow::Result<Return> {
        self.post("/returns", &request).await
    }

    async fn get_return(&self, return_id: &str) -> anyhow::Result<Return> {
        self.get(&format!("/returns/{}", return_id)).await
    }

    async fn list_returns(&self, params: ListReturnsParams) -> anyhow::Result<ReturnList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/returns".to_string()
        } else {
            format!("/returns?{}", query)
        };
        self.get(&path).await
    }

    async fn approve_return(
        &self,
        return_id: &str,
        refund_amount: f64,
        notes: Option<String>,
    ) -> anyhow::Result<Return> {
        #[derive(Serialize)]
        struct ApproveRequest {
            refund_amount: f64,
            #[serde(skip_serializing_if = "Option::is_none")]
            notes: Option<String>,
        }

        self.post(
            &format!("/returns/{}/approve", return_id),
            &ApproveRequest { refund_amount, notes },
        ).await
    }

    async fn reject_return(&self, return_id: &str, reason: &str) -> anyhow::Result<Return> {
        #[derive(Serialize)]
        struct RejectRequest {
            reason: String,
        }

        self.post(
            &format!("/returns/{}/reject", return_id),
            &RejectRequest { reason: reason.to_string() },
        ).await
    }

    // ========================================================================
    // Shipments
    // ========================================================================

    async fn create_shipment(&self, request: CreateShipmentRequest) -> anyhow::Result<Shipment> {
        self.post("/shipments", &request).await
    }

    async fn get_shipment(&self, shipment_id: &str) -> anyhow::Result<Shipment> {
        self.get(&format!("/shipments/{}", shipment_id)).await
    }

    async fn list_shipments(&self, params: ListShipmentsParams) -> anyhow::Result<ShipmentList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/shipments".to_string()
        } else {
            format!("/shipments?{}", query)
        };
        self.get(&path).await
    }

    async fn update_shipment_status(
        &self,
        shipment_id: &str,
        status: ShipmentStatus,
        tracking: Option<String>,
    ) -> anyhow::Result<Shipment> {
        #[derive(Serialize)]
        struct UpdateRequest {
            status: ShipmentStatus,
            #[serde(skip_serializing_if = "Option::is_none")]
            tracking_number: Option<String>,
        }

        self.put(
            &format!("/shipments/{}/status", shipment_id),
            &UpdateRequest {
                status,
                tracking_number: tracking,
            },
        ).await
    }

    // ========================================================================
    // Customers
    // ========================================================================

    async fn get_customer(&self, customer_id: &str) -> anyhow::Result<Customer> {
        self.get(&format!("/customers/{}", customer_id)).await
    }

    async fn list_customers(&self, params: ListCustomersParams) -> anyhow::Result<CustomerList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/customers".to_string()
        } else {
            format!("/customers?{}", query)
        };
        self.get(&path).await
    }

    async fn get_customer_orders(&self, customer_id: &str) -> anyhow::Result<OrderList> {
        self.get(&format!("/customers/{}/orders", customer_id)).await
    }

    // ========================================================================
    // Products
    // ========================================================================

    async fn get_product(&self, product_id: &str) -> anyhow::Result<Product> {
        self.get(&format!("/products/{}", product_id)).await
    }

    async fn list_products(&self, params: ListProductsParams) -> anyhow::Result<ProductList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/products".to_string()
        } else {
            format!("/products?{}", query)
        };
        self.get(&path).await
    }

    async fn search_products(&self, query: &str, limit: Option<u32>) -> anyhow::Result<Vec<Product>> {
        let limit = limit.unwrap_or(20);
        let path = format!("/products/search?q={}&limit={}", urlencoding::encode(query), limit);
        let list: ProductList = self.get(&path).await?;
        Ok(list.items)
    }

    // ========================================================================
    // Purchase Orders
    // ========================================================================

    async fn create_purchase_order(&self, request: CreatePurchaseOrderRequest) -> anyhow::Result<PurchaseOrder> {
        self.post("/purchase-orders", &request).await
    }

    async fn get_purchase_order(&self, po_id: &str) -> anyhow::Result<PurchaseOrder> {
        self.get(&format!("/purchase-orders/{}", po_id)).await
    }

    async fn list_purchase_orders(&self, params: ListPurchaseOrdersParams) -> anyhow::Result<PurchaseOrderList> {
        let query = self.build_query(&params);
        let path = if query.is_empty() {
            "/purchase-orders".to_string()
        } else {
            format!("/purchase-orders?{}", query)
        };
        self.get(&path).await
    }

    // ========================================================================
    // Analytics
    // ========================================================================

    async fn get_dashboard_metrics(&self) -> anyhow::Result<DashboardMetrics> {
        self.get("/analytics/dashboard").await
    }

    async fn get_sales_trends(
        &self,
        start_date: &str,
        end_date: &str,
        interval: &str,
    ) -> anyhow::Result<Vec<SalesTrend>> {
        let path = format!(
            "/analytics/sales/trends?start_date={}&end_date={}&interval={}",
            start_date, end_date, interval
        );
        self.get(&path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = ApiConfig {
            rest_url: "http://localhost:8080/api/v1".to_string(),
            grpc_url: None,
            api_key: "test-key".to_string(),
            timeout_secs: 30,
            max_retries: 3,
            prefer_grpc: false,
        };

        let client = RestClient::new(&config).unwrap();
        assert_eq!(client.base_url, "http://localhost:8080/api/v1");
    }
}
