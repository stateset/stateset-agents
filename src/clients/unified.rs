//! Unified StateSet Client
//!
//! Provides a single client interface that automatically selects between REST and gRPC
//! based on configuration and availability.

use super::*;
use super::rest::RestClient;
use super::grpc::GrpcClient;
use crate::config::AgentConfig;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Unified client that can use either REST or gRPC
pub struct StateSetClient {
    rest_client: RestClient,
    grpc_client: Option<GrpcClient>,
    prefer_grpc: bool,
}

impl StateSetClient {
    /// Create a new unified client from configuration
    pub async fn new(config: &AgentConfig) -> anyhow::Result<Self> {
        let rest_client = RestClient::new(&config.api)?;
        info!("REST client initialized for {}", config.api.rest_url);

        let grpc_client = if let Some(ref grpc_url) = config.api.grpc_url {
            match GrpcClient::new(&config.api).await {
                Ok(client) => {
                    info!("gRPC client initialized for {}", grpc_url);
                    Some(client)
                }
                Err(e) => {
                    warn!("Failed to initialize gRPC client: {}. Falling back to REST only.", e);
                    None
                }
            }
        } else {
            debug!("gRPC URL not configured, using REST only");
            None
        };

        Ok(Self {
            rest_client,
            grpc_client,
            prefer_grpc: config.api.prefer_grpc,
        })
    }

    /// Create a REST-only client
    pub fn rest_only(rest_url: &str, api_key: &str) -> anyhow::Result<Self> {
        let rest_client = RestClient::with_params(rest_url, api_key, 30)?;

        Ok(Self {
            rest_client,
            grpc_client: None,
            prefer_grpc: false,
        })
    }

    /// Check if gRPC is available
    pub fn has_grpc(&self) -> bool {
        self.grpc_client.is_some()
    }

    /// Get the underlying REST client
    pub fn rest(&self) -> &RestClient {
        &self.rest_client
    }

    /// Get the underlying gRPC client (if available)
    pub fn grpc(&self) -> Option<&GrpcClient> {
        self.grpc_client.as_ref()
    }

    /// Choose the appropriate client for an operation
    fn choose_client(&self) -> &dyn CommerceApi {
        if self.prefer_grpc && self.grpc_client.is_some() {
            // gRPC is preferred and available
            // Note: In practice, check if the specific operation is implemented
            self.grpc_client.as_ref().unwrap() as &dyn CommerceApi
        } else {
            &self.rest_client as &dyn CommerceApi
        }
    }
}

/// The unified client delegates to the appropriate underlying client
#[async_trait]
impl CommerceApi for StateSetClient {
    // Orders
    async fn create_order(&self, request: CreateOrderRequest) -> anyhow::Result<Order> {
        // Always use REST for now since gRPC is not fully implemented
        self.rest_client.create_order(request).await
    }

    async fn get_order(&self, order_id: &str) -> anyhow::Result<Order> {
        self.rest_client.get_order(order_id).await
    }

    async fn list_orders(&self, params: ListOrdersParams) -> anyhow::Result<OrderList> {
        self.rest_client.list_orders(params).await
    }

    async fn update_order_status(
        &self,
        order_id: &str,
        status: OrderStatus,
        notes: Option<String>,
    ) -> anyhow::Result<Order> {
        self.rest_client.update_order_status(order_id, status, notes).await
    }

    async fn cancel_order(&self, order_id: &str, reason: &str) -> anyhow::Result<Order> {
        self.rest_client.cancel_order(order_id, reason).await
    }

    // Inventory
    async fn get_inventory(&self, product_id: &str, location_id: Option<&str>) -> anyhow::Result<InventoryItem> {
        self.rest_client.get_inventory(product_id, location_id).await
    }

    async fn list_inventory(&self, params: ListInventoryParams) -> anyhow::Result<InventoryList> {
        self.rest_client.list_inventory(params).await
    }

    async fn reserve_inventory(&self, request: ReserveInventoryRequest) -> anyhow::Result<InventoryReservation> {
        self.rest_client.reserve_inventory(request).await
    }

    async fn release_inventory(&self, reservation_id: &str) -> anyhow::Result<()> {
        self.rest_client.release_inventory(reservation_id).await
    }

    async fn adjust_inventory(&self, request: AdjustInventoryRequest) -> anyhow::Result<InventoryItem> {
        self.rest_client.adjust_inventory(request).await
    }

    async fn get_low_stock(&self) -> anyhow::Result<Vec<InventoryItem>> {
        self.rest_client.get_low_stock().await
    }

    // Returns
    async fn create_return(&self, request: CreateReturnRequest) -> anyhow::Result<Return> {
        self.rest_client.create_return(request).await
    }

    async fn get_return(&self, return_id: &str) -> anyhow::Result<Return> {
        self.rest_client.get_return(return_id).await
    }

    async fn list_returns(&self, params: ListReturnsParams) -> anyhow::Result<ReturnList> {
        self.rest_client.list_returns(params).await
    }

    async fn approve_return(
        &self,
        return_id: &str,
        refund_amount: f64,
        notes: Option<String>,
    ) -> anyhow::Result<Return> {
        self.rest_client.approve_return(return_id, refund_amount, notes).await
    }

    async fn reject_return(&self, return_id: &str, reason: &str) -> anyhow::Result<Return> {
        self.rest_client.reject_return(return_id, reason).await
    }

    // Shipments
    async fn create_shipment(&self, request: CreateShipmentRequest) -> anyhow::Result<Shipment> {
        self.rest_client.create_shipment(request).await
    }

    async fn get_shipment(&self, shipment_id: &str) -> anyhow::Result<Shipment> {
        self.rest_client.get_shipment(shipment_id).await
    }

    async fn list_shipments(&self, params: ListShipmentsParams) -> anyhow::Result<ShipmentList> {
        self.rest_client.list_shipments(params).await
    }

    async fn update_shipment_status(
        &self,
        shipment_id: &str,
        status: ShipmentStatus,
        tracking: Option<String>,
    ) -> anyhow::Result<Shipment> {
        self.rest_client.update_shipment_status(shipment_id, status, tracking).await
    }

    // Customers
    async fn get_customer(&self, customer_id: &str) -> anyhow::Result<Customer> {
        self.rest_client.get_customer(customer_id).await
    }

    async fn list_customers(&self, params: ListCustomersParams) -> anyhow::Result<CustomerList> {
        self.rest_client.list_customers(params).await
    }

    async fn get_customer_orders(&self, customer_id: &str) -> anyhow::Result<OrderList> {
        self.rest_client.get_customer_orders(customer_id).await
    }

    // Products
    async fn get_product(&self, product_id: &str) -> anyhow::Result<Product> {
        self.rest_client.get_product(product_id).await
    }

    async fn list_products(&self, params: ListProductsParams) -> anyhow::Result<ProductList> {
        self.rest_client.list_products(params).await
    }

    async fn search_products(&self, query: &str, limit: Option<u32>) -> anyhow::Result<Vec<Product>> {
        self.rest_client.search_products(query, limit).await
    }

    // Purchase Orders
    async fn create_purchase_order(&self, request: CreatePurchaseOrderRequest) -> anyhow::Result<PurchaseOrder> {
        self.rest_client.create_purchase_order(request).await
    }

    async fn get_purchase_order(&self, po_id: &str) -> anyhow::Result<PurchaseOrder> {
        self.rest_client.get_purchase_order(po_id).await
    }

    async fn list_purchase_orders(&self, params: ListPurchaseOrdersParams) -> anyhow::Result<PurchaseOrderList> {
        self.rest_client.list_purchase_orders(params).await
    }

    // Analytics
    async fn get_dashboard_metrics(&self) -> anyhow::Result<DashboardMetrics> {
        self.rest_client.get_dashboard_metrics().await
    }

    async fn get_sales_trends(
        &self,
        start_date: &str,
        end_date: &str,
        interval: &str,
    ) -> anyhow::Result<Vec<SalesTrend>> {
        self.rest_client.get_sales_trends(start_date, end_date, interval).await
    }
}

// ============================================================================
// Client Builder Pattern
// ============================================================================

/// Builder for creating StateSet clients with custom configuration
pub struct StateSetClientBuilder {
    rest_url: Option<String>,
    grpc_url: Option<String>,
    api_key: Option<String>,
    timeout_secs: u64,
    max_retries: u32,
    prefer_grpc: bool,
}

impl Default for StateSetClientBuilder {
    fn default() -> Self {
        Self {
            rest_url: None,
            grpc_url: None,
            api_key: None,
            timeout_secs: 30,
            max_retries: 3,
            prefer_grpc: false,
        }
    }
}

impl StateSetClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rest_url(mut self, url: &str) -> Self {
        self.rest_url = Some(url.to_string());
        self
    }

    pub fn grpc_url(mut self, url: &str) -> Self {
        self.grpc_url = Some(url.to_string());
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    pub fn timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn prefer_grpc(mut self, prefer: bool) -> Self {
        self.prefer_grpc = prefer;
        self
    }

    pub async fn build(self) -> anyhow::Result<StateSetClient> {
        let rest_url = self.rest_url
            .ok_or_else(|| anyhow::anyhow!("REST URL is required"))?;
        let api_key = self.api_key
            .ok_or_else(|| anyhow::anyhow!("API key is required"))?;

        let api_config = crate::config::ApiConfig {
            rest_url,
            grpc_url: self.grpc_url,
            api_key,
            timeout_secs: self.timeout_secs,
            max_retries: self.max_retries,
            prefer_grpc: self.prefer_grpc,
        };

        let rest_client = RestClient::new(&api_config)?;

        let grpc_client = if api_config.grpc_url.is_some() {
            GrpcClient::new(&api_config).await.ok()
        } else {
            None
        };

        Ok(StateSetClient {
            rest_client,
            grpc_client,
            prefer_grpc: self.prefer_grpc,
        })
    }
}
