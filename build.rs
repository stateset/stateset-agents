fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile proto files for gRPC client generation
    // These should match the StateSet API proto definitions

    let proto_files = [
        "../stateset-api/proto/common.proto",
        "../stateset-api/proto/order.proto",
        "../stateset-api/proto/inventory.proto",
        "../stateset-api/proto/return.proto",
        "../stateset-api/proto/shipment.proto",
        "../stateset-api/proto/customer.proto",
        "../stateset-api/proto/product.proto",
        "../stateset-api/proto/purchase_order.proto",
        "../stateset-api/proto/work_order.proto",
        "../stateset-api/proto/warranty.proto",
    ];

    let include_dirs = [
        "../stateset-api/proto",
        "../stateset-api/include",
    ];

    // Only compile if proto files exist
    let protos_exist = proto_files.iter().all(|p| std::path::Path::new(p).exists());

    if protos_exist {
        tonic_build::configure()
            .build_server(false) // We only need client stubs
            .build_client(true)
            .out_dir("src/proto")
            .compile_protos(&proto_files, &include_dirs)?;

        println!("cargo:rerun-if-changed=../stateset-api/proto/");
    } else {
        println!("cargo:warning=Proto files not found, skipping gRPC codegen. REST client will still work.");
    }

    Ok(())
}
