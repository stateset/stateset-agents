//! API Clients for StateSet API
//!
//! Provides both REST and gRPC clients that can be used interchangeably.

pub mod rest;
pub mod grpc;
pub mod unified;

pub use rest::RestClient;
pub use unified::StateSetClient;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Common trait for all API operations
/// Allows switching between REST and gRPC transparently
#[async_trait]
pub trait CommerceApi: Send + Sync {
    // Orders
    async fn create_order(&self, request: CreateOrderRequest) -> anyhow::Result<Order>;
    async fn get_order(&self, order_id: &str) -> anyhow::Result<Order>;
    async fn list_orders(&self, params: ListOrdersParams) -> anyhow::Result<OrderList>;
    async fn update_order_status(&self, order_id: &str, status: OrderStatus, notes: Option<String>) -> anyhow::Result<Order>;
    async fn cancel_order(&self, order_id: &str, reason: &str) -> anyhow::Result<Order>;

    // Inventory
    async fn get_inventory(&self, product_id: &str, location_id: Option<&str>) -> anyhow::Result<InventoryItem>;
    async fn list_inventory(&self, params: ListInventoryParams) -> anyhow::Result<InventoryList>;
    async fn reserve_inventory(&self, request: ReserveInventoryRequest) -> anyhow::Result<InventoryReservation>;
    async fn release_inventory(&self, reservation_id: &str) -> anyhow::Result<()>;
    async fn adjust_inventory(&self, request: AdjustInventoryRequest) -> anyhow::Result<InventoryItem>;
    async fn get_low_stock(&self) -> anyhow::Result<Vec<InventoryItem>>;

    // Returns
    async fn create_return(&self, request: CreateReturnRequest) -> anyhow::Result<Return>;
    async fn get_return(&self, return_id: &str) -> anyhow::Result<Return>;
    async fn list_returns(&self, params: ListReturnsParams) -> anyhow::Result<ReturnList>;
    async fn approve_return(&self, return_id: &str, refund_amount: f64, notes: Option<String>) -> anyhow::Result<Return>;
    async fn reject_return(&self, return_id: &str, reason: &str) -> anyhow::Result<Return>;

    // Shipments
    async fn create_shipment(&self, request: CreateShipmentRequest) -> anyhow::Result<Shipment>;
    async fn get_shipment(&self, shipment_id: &str) -> anyhow::Result<Shipment>;
    async fn list_shipments(&self, params: ListShipmentsParams) -> anyhow::Result<ShipmentList>;
    async fn update_shipment_status(&self, shipment_id: &str, status: ShipmentStatus, tracking: Option<String>) -> anyhow::Result<Shipment>;

    // Customers
    async fn get_customer(&self, customer_id: &str) -> anyhow::Result<Customer>;
    async fn list_customers(&self, params: ListCustomersParams) -> anyhow::Result<CustomerList>;
    async fn get_customer_orders(&self, customer_id: &str) -> anyhow::Result<OrderList>;

    // Products
    async fn get_product(&self, product_id: &str) -> anyhow::Result<Product>;
    async fn list_products(&self, params: ListProductsParams) -> anyhow::Result<ProductList>;
    async fn search_products(&self, query: &str, limit: Option<u32>) -> anyhow::Result<Vec<Product>>;

    // Purchase Orders
    async fn create_purchase_order(&self, request: CreatePurchaseOrderRequest) -> anyhow::Result<PurchaseOrder>;
    async fn get_purchase_order(&self, po_id: &str) -> anyhow::Result<PurchaseOrder>;
    async fn list_purchase_orders(&self, params: ListPurchaseOrdersParams) -> anyhow::Result<PurchaseOrderList>;

    // Analytics
    async fn get_dashboard_metrics(&self) -> anyhow::Result<DashboardMetrics>;
    async fn get_sales_trends(&self, start_date: &str, end_date: &str, interval: &str) -> anyhow::Result<Vec<SalesTrend>>;
}

// ============================================================================
// Data Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub order_number: Option<String>,
    pub customer_id: String,
    pub status: OrderStatus,
    pub items: Vec<OrderItem>,
    pub total_amount: f64,
    pub currency: String,
    pub shipping_address: Option<Address>,
    pub billing_address: Option<Address>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderItem {
    pub id: String,
    pub product_id: String,
    pub sku: Option<String>,
    pub name: String,
    pub quantity: i32,
    pub unit_price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    Pending,
    Processing,
    Shipped,
    Delivered,
    Completed,
    Canceled,
    Returned,
    OnHold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Address {
    pub name: Option<String>,
    pub line1: String,
    pub line2: Option<String>,
    pub city: String,
    pub state: Option<String>,
    pub postal_code: String,
    pub country: String,
    pub phone: Option<String>,
    pub email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryItem {
    pub id: String,
    pub product_id: String,
    pub sku: String,
    pub location_id: String,
    pub quantity_on_hand: i32,
    pub quantity_reserved: i32,
    pub quantity_available: i32,
    pub reorder_point: Option<i32>,
    pub safety_stock: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryReservation {
    pub id: String,
    pub inventory_id: String,
    pub quantity: i32,
    pub order_id: Option<String>,
    pub expires_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Return {
    pub id: String,
    pub rma_number: Option<String>,
    pub order_id: String,
    pub status: ReturnStatus,
    pub items: Vec<ReturnItem>,
    pub reason: String,
    pub customer_notes: Option<String>,
    pub refund_amount: Option<f64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnItem {
    pub order_item_id: String,
    pub quantity: i32,
    pub reason: String,
    pub condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReturnStatus {
    Pending,
    Approved,
    Rejected,
    Received,
    Restocked,
    Refunded,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shipment {
    pub id: String,
    pub order_id: String,
    pub status: ShipmentStatus,
    pub carrier: Option<String>,
    pub tracking_number: Option<String>,
    pub shipped_at: Option<String>,
    pub delivered_at: Option<String>,
    pub items: Vec<ShipmentItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipmentItem {
    pub order_item_id: String,
    pub quantity: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShipmentStatus {
    Pending,
    Processing,
    Shipped,
    InTransit,
    OutForDelivery,
    Delivered,
    Failed,
    Returned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Customer {
    pub id: String,
    pub email: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub phone: Option<String>,
    pub addresses: Vec<Address>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Product {
    pub id: String,
    pub name: String,
    pub sku: String,
    pub description: Option<String>,
    pub price: f64,
    pub currency: String,
    pub category: Option<String>,
    pub inventory_quantity: Option<i32>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurchaseOrder {
    pub id: String,
    pub po_number: Option<String>,
    pub supplier_id: String,
    pub status: PurchaseOrderStatus,
    pub items: Vec<PurchaseOrderItem>,
    pub total_amount: f64,
    pub expected_delivery: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurchaseOrderItem {
    pub product_id: String,
    pub quantity: i32,
    pub unit_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PurchaseOrderStatus {
    Draft,
    Submitted,
    Confirmed,
    Shipped,
    Received,
    Canceled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub total_orders: i64,
    pub total_revenue: f64,
    pub average_order_value: f64,
    pub orders_today: i64,
    pub revenue_today: f64,
    pub low_stock_items: i64,
    pub pending_shipments: i64,
    pub pending_returns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalesTrend {
    pub period: String,
    pub order_count: i64,
    pub revenue: f64,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateOrderRequest {
    pub customer_id: String,
    pub items: Vec<OrderItemRequest>,
    pub shipping_address: Option<Address>,
    pub billing_address: Option<Address>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderItemRequest {
    pub product_id: String,
    pub sku: Option<String>,
    pub quantity: i32,
    pub unit_price: f64,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListOrdersParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub status: Option<OrderStatus>,
    pub customer_id: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderList {
    pub items: Vec<Order>,
    pub total: i64,
    pub page: u32,
    pub limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListInventoryParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub product_id: Option<String>,
    pub location_id: Option<String>,
    pub low_stock: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryList {
    pub items: Vec<InventoryItem>,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReserveInventoryRequest {
    pub inventory_id: String,
    pub quantity: i32,
    pub order_id: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustInventoryRequest {
    pub inventory_id: String,
    pub quantity_change: i32,
    pub reason: String,
    pub reference_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateReturnRequest {
    pub order_id: String,
    pub items: Vec<ReturnItemRequest>,
    pub reason: String,
    pub customer_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnItemRequest {
    pub order_item_id: String,
    pub quantity: i32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListReturnsParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub status: Option<ReturnStatus>,
    pub order_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnList {
    pub items: Vec<Return>,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateShipmentRequest {
    pub order_id: String,
    pub carrier: Option<String>,
    pub service_level: Option<String>,
    pub items: Vec<ShipmentItemRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipmentItemRequest {
    pub order_item_id: String,
    pub quantity: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListShipmentsParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub status: Option<ShipmentStatus>,
    pub order_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipmentList {
    pub items: Vec<Shipment>,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListCustomersParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub search: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerList {
    pub items: Vec<Customer>,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListProductsParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub category: Option<String>,
    pub active: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductList {
    pub items: Vec<Product>,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePurchaseOrderRequest {
    pub supplier_id: String,
    pub items: Vec<PurchaseOrderItemRequest>,
    pub notes: Option<String>,
    pub expected_delivery: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurchaseOrderItemRequest {
    pub product_id: String,
    pub quantity: i32,
    pub unit_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListPurchaseOrdersParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub status: Option<PurchaseOrderStatus>,
    pub supplier_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurchaseOrderList {
    pub items: Vec<PurchaseOrder>,
    pub total: i64,
}
