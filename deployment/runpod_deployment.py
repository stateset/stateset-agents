"""
RunPod Cloud Deployment for GRPO Agent Framework

This module provides automated deployment and management of GRPO training
jobs on RunPod cloud infrastructure with dynamic resource scaling.
"""

import asyncio
import base64
import io
import json
import logging
import os
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import runpod

    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False

from ..training.config import TrainingConfig
from ..training.distributed_trainer import DistributedConfig
from ..utils.monitoring import MonitoringService

logger = logging.getLogger(__name__)


@dataclass
class RunPodConfig:
    """Configuration for RunPod deployment"""

    # API Configuration
    api_key: str = ""
    endpoint_id: str = ""

    # Pod Configuration
    pod_type: str = "NVIDIA RTX 4090"  # GPU type
    image_name: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
    container_disk_size: int = 20  # GB
    volume_size: int = 100  # GB

    # Environment
    environment_variables: Dict[str, str] = field(default_factory=dict)
    ports: List[str] = field(default_factory=lambda: ["8888/http", "6006/http"])

    # Scaling
    min_pods: int = 1
    max_pods: int = 8
    scale_threshold: float = 0.8  # GPU utilization threshold

    # Deployment
    deployment_timeout: int = 600  # seconds
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": "***",  # Don't expose API key
            "endpoint_id": self.endpoint_id,
            "pod_type": self.pod_type,
            "image_name": self.image_name,
            "container_disk_size": self.container_disk_size,
            "volume_size": self.volume_size,
            "environment_variables": self.environment_variables,
            "ports": self.ports,
            "min_pods": self.min_pods,
            "max_pods": self.max_pods,
            "scale_threshold": self.scale_threshold,
            "deployment_timeout": self.deployment_timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class DeploymentStatus:
    """Status of a RunPod deployment"""

    deployment_id: str
    status: str
    pods: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "status": self.status,
            "pods": self.pods,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metrics": self.metrics,
            "logs": self.logs,
        }


class RunPodDeploymentManager:
    """
    Manager for RunPod cloud deployments
    """

    def __init__(
        self,
        config: RunPodConfig,
        monitoring_service: Optional[MonitoringService] = None,
    ):
        if not RUNPOD_AVAILABLE:
            raise ImportError(
                "runpod package is required. Install with: pip install runpod"
            )

        self.config = config
        self.monitoring = monitoring_service

        # Initialize RunPod client
        runpod.api_key = self.config.api_key

        # Deployment tracking
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.active_pods: Dict[str, Dict[str, Any]] = {}

        # Metrics
        self.deployment_count = 0
        self.total_cost = 0.0

    async def deploy_training_job(
        self,
        training_config: TrainingConfig,
        distributed_config: DistributedConfig,
        code_bundle: Optional[str] = None,
        requirements: Optional[List[str]] = None,
    ) -> str:
        """
        Deploy a GRPO training job to RunPod

        Args:
            training_config: GRPO training configuration
            distributed_config: Distributed training configuration
            code_bundle: Path to code bundle or base64 encoded content
            requirements: List of Python requirements

        Returns:
            Deployment ID
        """
        deployment_id = f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create deployment status
        self.deployments[deployment_id] = DeploymentStatus(
            deployment_id=deployment_id, status="initializing"
        )

        try:
            # Prepare deployment configuration
            deployment_config = await self._prepare_deployment_config(
                training_config, distributed_config, code_bundle, requirements
            )

            # Launch pods
            pods = await self._launch_pods(
                deployment_config, distributed_config.world_size
            )

            # Update deployment status
            self.deployments[deployment_id].pods = pods
            self.deployments[deployment_id].status = "running"
            self.deployments[deployment_id].updated_at = datetime.now()

            # Start monitoring
            asyncio.create_task(self._monitor_deployment(deployment_id))

            self.deployment_count += 1

            logger.info(f"Deployed training job: {deployment_id}")

            return deployment_id

        except Exception as e:
            self.deployments[deployment_id].status = "failed"
            self.deployments[deployment_id].logs.append(f"Deployment failed: {str(e)}")
            logger.error(f"Failed to deploy training job: {e}")
            raise

    async def _prepare_deployment_config(
        self,
        training_config: TrainingConfig,
        distributed_config: DistributedConfig,
        code_bundle: Optional[str] = None,
        requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Prepare deployment configuration"""

        # Environment variables
        env_vars = {
            "GRPO_TRAINING_CONFIG": json.dumps(training_config.to_dict()),
            "GRPO_DISTRIBUTED_CONFIG": json.dumps(distributed_config.to_dict()),
            "PYTHONPATH": "/workspace",
            "CUDA_VISIBLE_DEVICES": "0",
            **self.config.environment_variables,
        }

        # Docker commands
        docker_commands = [
            "cd /workspace",
            "pip install -r requirements.txt" if requirements else "",
            "python -m stateset_agents.training.distributed_trainer",
        ]

        # Filter out empty commands
        docker_commands = [cmd for cmd in docker_commands if cmd]

        return {
            "name": f"grpo-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "image_name": self.config.image_name,
            "gpu_type_id": self.config.pod_type,
            "container_disk_in_gb": self.config.container_disk_size,
            "volume_in_gb": self.config.volume_size,
            "ports": self.config.ports,
            "env": env_vars,
            "docker_args": " && ".join(docker_commands),
            "code_bundle": code_bundle,
            "requirements": requirements or [],
        }

    async def _launch_pods(
        self, deployment_config: Dict[str, Any], num_pods: int
    ) -> List[Dict[str, Any]]:
        """Launch RunPod instances"""
        pods = []

        for i in range(num_pods):
            # Update environment for distributed training
            pod_config = deployment_config.copy()
            pod_config["env"]["RANK"] = str(i)
            pod_config["env"]["WORLD_SIZE"] = str(num_pods)
            pod_config["env"][
                "MASTER_ADDR"
            ] = "127.0.0.1"  # Will be updated after first pod
            pod_config["env"]["MASTER_PORT"] = "12355"

            # Launch pod
            pod = await self._launch_single_pod(pod_config)
            pods.append(pod)

            # Update master address for subsequent pods
            if i == 0 and "ip" in pod:
                deployment_config["env"]["MASTER_ADDR"] = pod["ip"]

        return pods

    async def _launch_single_pod(self, pod_config: Dict[str, Any]) -> Dict[str, Any]:
        """Launch a single RunPod instance"""
        for attempt in range(self.config.max_retries):
            try:
                # Create pod
                pod = runpod.create_pod(
                    name=pod_config["name"],
                    image_name=pod_config["image_name"],
                    gpu_type_id=pod_config["gpu_type_id"],
                    container_disk_in_gb=pod_config["container_disk_in_gb"],
                    volume_in_gb=pod_config["volume_in_gb"],
                    ports=pod_config["ports"],
                    env=pod_config["env"],
                    docker_args=pod_config["docker_args"],
                )

                # Wait for pod to be ready
                await self._wait_for_pod_ready(pod["id"])

                # Get pod details
                pod_details = runpod.get_pod(pod["id"])

                return {
                    "id": pod["id"],
                    "name": pod_config["name"],
                    "status": "running",
                    "ip": pod_details.get("runtime", {})
                    .get("ports", {})
                    .get("22/tcp", [{}])[0]
                    .get("ip"),
                    "created_at": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.warning(f"Pod launch attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(5)

        raise RuntimeError("Failed to launch pod after all retries")

    async def _wait_for_pod_ready(self, pod_id: str, timeout: int = 300):
        """Wait for pod to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            pod = runpod.get_pod(pod_id)

            if (
                pod.get("desiredStatus") == "RUNNING"
                and pod.get("runtime", {}).get("uptimeInSeconds", 0) > 0
            ):
                logger.info(f"Pod {pod_id} is ready")
                return

            await asyncio.sleep(10)

        raise TimeoutError(
            f"Pod {pod_id} failed to become ready within {timeout} seconds"
        )

    async def _monitor_deployment(self, deployment_id: str):
        """Monitor deployment status and metrics"""
        while self.deployments[deployment_id].status == "running":
            try:
                # Update pod statuses
                deployment = self.deployments[deployment_id]

                for pod in deployment.pods:
                    pod_details = runpod.get_pod(pod["id"])
                    pod["status"] = pod_details.get("desiredStatus", "unknown")

                    # Collect metrics
                    metrics = await self._collect_pod_metrics(pod["id"])
                    if metrics:
                        deployment.metrics[pod["id"]] = metrics

                # Check for completion or failure
                pod_statuses = [pod["status"] for pod in deployment.pods]

                if all(status == "EXITED" for status in pod_statuses):
                    deployment.status = "completed"
                    logger.info(f"Deployment {deployment_id} completed")
                    break
                elif any(status == "FAILED" for status in pod_statuses):
                    deployment.status = "failed"
                    logger.error(f"Deployment {deployment_id} failed")
                    break

                deployment.updated_at = datetime.now()

                # Log metrics
                if self.monitoring:
                    await self._log_deployment_metrics(deployment_id)

            except Exception as e:
                logger.error(f"Error monitoring deployment {deployment_id}: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _collect_pod_metrics(self, pod_id: str) -> Optional[Dict[str, Any]]:
        """Collect metrics from a pod"""
        try:
            pod = runpod.get_pod(pod_id)
            runtime = pod.get("runtime", {})

            return {
                "uptime": runtime.get("uptimeInSeconds", 0),
                "gpu_utilization": runtime.get("gpuUtilization", 0),
                "memory_utilization": runtime.get("memoryUtilization", 0),
                "cpu_utilization": runtime.get("cpuUtilization", 0),
                "cost_per_hour": pod.get("costPerHr", 0),
                "total_cost": pod.get("totalCost", 0),
            }
        except Exception as e:
            logger.error(f"Failed to collect metrics for pod {pod_id}: {e}")
            return None

    async def _log_deployment_metrics(self, deployment_id: str):
        """Log deployment metrics to monitoring service"""
        deployment = self.deployments[deployment_id]

        # Aggregate metrics
        total_cost = sum(
            pod_metrics.get("total_cost", 0)
            for pod_metrics in deployment.metrics.values()
        )

        avg_gpu_utilization = (
            sum(
                pod_metrics.get("gpu_utilization", 0)
                for pod_metrics in deployment.metrics.values()
            )
            / len(deployment.metrics)
            if deployment.metrics
            else 0
        )

        # Log metrics
        await self.monitoring.log_metric("runpod.deployment.cost", total_cost)
        await self.monitoring.log_metric(
            "runpod.deployment.gpu_utilization", avg_gpu_utilization
        )
        await self.monitoring.log_metric(
            "runpod.deployment.num_pods", len(deployment.pods)
        )

    async def scale_deployment(
        self, deployment_id: str, target_pods: int
    ) -> Dict[str, Any]:
        """Scale deployment up or down"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.deployments[deployment_id]
        current_pods = len(deployment.pods)

        if target_pods == current_pods:
            return {"message": "No scaling needed", "current_pods": current_pods}

        if target_pods > current_pods:
            # Scale up
            new_pods = await self._scale_up(deployment, target_pods - current_pods)
            deployment.pods.extend(new_pods)

        else:
            # Scale down
            pods_to_remove = current_pods - target_pods
            await self._scale_down(deployment, pods_to_remove)
            deployment.pods = deployment.pods[:target_pods]

        deployment.updated_at = datetime.now()

        return {
            "message": f"Scaled deployment from {current_pods} to {target_pods} pods",
            "previous_pods": current_pods,
            "current_pods": target_pods,
        }

    async def _scale_up(
        self, deployment: DeploymentStatus, num_pods: int
    ) -> List[Dict[str, Any]]:
        """Scale up deployment"""
        # Get base configuration from existing pods
        if not deployment.pods:
            raise ValueError("Cannot scale up without existing pods")

        base_config = {
            "name": f"grpo-scale-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "image_name": self.config.image_name,
            "gpu_type_id": self.config.pod_type,
            "container_disk_in_gb": self.config.container_disk_size,
            "volume_in_gb": self.config.volume_size,
            "ports": self.config.ports,
            "env": self.config.environment_variables,
        }

        new_pods = []
        for i in range(num_pods):
            pod = await self._launch_single_pod(base_config)
            new_pods.append(pod)

        return new_pods

    async def _scale_down(self, deployment: DeploymentStatus, num_pods: int):
        """Scale down deployment"""
        pods_to_remove = deployment.pods[-num_pods:]

        for pod in pods_to_remove:
            try:
                runpod.terminate_pod(pod["id"])
                logger.info(f"Terminated pod {pod['id']}")
            except Exception as e:
                logger.error(f"Failed to terminate pod {pod['id']}: {e}")

    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        return self.deployments[deployment_id].to_dict()

    async def get_deployment_logs(self, deployment_id: str) -> List[str]:
        """Get deployment logs"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.deployments[deployment_id]
        all_logs = []

        for pod in deployment.pods:
            try:
                pod_logs = runpod.get_pod_logs(pod["id"])
                all_logs.extend([f"[{pod['id']}] {log}" for log in pod_logs])
            except Exception as e:
                all_logs.append(f"[{pod['id']}] Failed to get logs: {e}")

        return all_logs

    async def terminate_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Terminate a deployment"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.deployments[deployment_id]

        # Terminate all pods
        terminated_pods = []
        for pod in deployment.pods:
            try:
                runpod.terminate_pod(pod["id"])
                terminated_pods.append(pod["id"])
                logger.info(f"Terminated pod {pod['id']}")
            except Exception as e:
                logger.error(f"Failed to terminate pod {pod['id']}: {e}")

        # Update deployment status
        deployment.status = "terminated"
        deployment.updated_at = datetime.now()

        return {
            "message": f"Terminated deployment {deployment_id}",
            "terminated_pods": terminated_pods,
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for all deployments"""
        total_cost = 0.0
        active_deployments = 0

        for deployment in self.deployments.values():
            if deployment.status == "running":
                active_deployments += 1

            for pod_metrics in deployment.metrics.values():
                total_cost += pod_metrics.get("total_cost", 0)

        return {
            "total_cost": total_cost,
            "active_deployments": active_deployments,
            "total_deployments": len(self.deployments),
            "cost_per_deployment": total_cost / len(self.deployments)
            if self.deployments
            else 0,
        }


# Utility functions
def create_code_bundle(source_dir: str, output_path: Optional[str] = None) -> str:
    """Create a code bundle for deployment"""
    if output_path is None:
        output_path = (
            f"/tmp/grpo_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        )

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=".")

    # Encode as base64 for API
    with open(output_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded


def get_runpod_config_from_env() -> RunPodConfig:
    """Create RunPod configuration from environment variables"""
    return RunPodConfig(
        api_key=os.getenv("RUNPOD_API_KEY", ""),
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID", ""),
        pod_type=os.getenv("RUNPOD_POD_TYPE", "NVIDIA RTX 4090"),
        image_name=os.getenv(
            "RUNPOD_IMAGE", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
        ),
        container_disk_size=int(os.getenv("RUNPOD_DISK_SIZE", "20")),
        volume_size=int(os.getenv("RUNPOD_VOLUME_SIZE", "100")),
        min_pods=int(os.getenv("RUNPOD_MIN_PODS", "1")),
        max_pods=int(os.getenv("RUNPOD_MAX_PODS", "8")),
    )


async def deploy_grpo_training(
    training_config: TrainingConfig,
    distributed_config: DistributedConfig,
    runpod_config: Optional[RunPodConfig] = None,
    source_dir: Optional[str] = None,
    requirements: Optional[List[str]] = None,
) -> str:
    """
    Convenience function to deploy GRPO training to RunPod

    Args:
        training_config: GRPO training configuration
        distributed_config: Distributed training configuration
        runpod_config: RunPod configuration (uses env vars if None)
        source_dir: Source code directory to bundle
        requirements: Python requirements

    Returns:
        Deployment ID
    """
    if runpod_config is None:
        runpod_config = get_runpod_config_from_env()

    # Create deployment manager
    manager = RunPodDeploymentManager(runpod_config)

    # Create code bundle if source directory provided
    code_bundle = None
    if source_dir:
        code_bundle = create_code_bundle(source_dir)

    # Deploy training job
    deployment_id = await manager.deploy_training_job(
        training_config=training_config,
        distributed_config=distributed_config,
        code_bundle=code_bundle,
        requirements=requirements,
    )

    return deployment_id
