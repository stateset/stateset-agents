"""
Setup configuration for GRPO Agent Framework
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stateset-agents",
    version="0.6.0",
    author="StateSet Team",
    author_email="team@stateset.ai",
    description="A comprehensive framework for training multi-turn AI agents using Group Relative Policy Optimization (GRPO)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stateset/stateset-agents",
    packages=find_packages(
        include=[
            # Primary namespace
            "stateset_agents",
            "stateset_agents.*",
            # Top-level implementation packages used by stateset_agents
            "core",
            "core.*",
            "training",
            "training.*",
            "utils",
            "utils.*",
            "rewards",
            "rewards.*",
            "api",
            "api.*",
            "grpo_agent_framework",
            "grpo_agent_framework.*",
            "environments",
            "environments.*",
        ]
    ),
    license="Business Source License 1.1",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "typing-extensions>=4.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "ruff>=0.1.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
            "semgrep>=1.0.0",
            "torch>=2.0.0",
            "transformers>=4.30.0,<4.45.0",
            "datasets>=2.0.0",
            "accelerate>=0.20.0",
            "wandb>=0.15.0",
            "peft>=0.4.0",
            "trl>=0.7.0",
            "aiohttp>=3.8.0",
            "psutil>=5.9.0",
            "scikit-learn>=1.3.0,<1.6.0",
        ],
        "api": [
            "fastapi>=0.110.0",
            "uvicorn>=0.23.0",
        ],
        "examples": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "langchain>=0.1.0",
        ],
        "training": [
            "torch>=2.0.0",
            "transformers>=4.30.0,<4.45.0",
            "datasets>=2.0.0",
            "accelerate>=0.20.0",
            "wandb>=0.15.0",
            "peft>=0.4.0",
            "trl>=0.7.0",
            "aiohttp>=3.8.0",
            "psutil>=5.9.0",
            "scikit-learn>=1.3.0,<1.6.0",
        ],
        "trl": [
            "trl>=0.7.0",
            "bitsandbytes>=0.41.0",  # For quantization support
        ],
        "hpo": [
            "optuna>=3.0.0",  # Primary HPO backend
            "plotly>=5.0.0",  # For Optuna visualizations
            "kaleido>=0.2.0",  # For saving plots
        ],
    },
    entry_points={
        "console_scripts": [
            "stateset-agents=stateset_agents.cli:run",
        ],
    },
)
