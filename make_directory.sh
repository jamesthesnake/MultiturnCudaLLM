#!/bin/bash

# Setup script for Kevin-Mirage-Optimization project
# This script creates the complete directory structure and touches all files

echo "Setting up Kevin-Mirage-Optimization project structure..."

# Create root directory
PROJECT_NAME="kevin-mirage-optimization"
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create main directories
echo "Creating main directories..."
mkdir -p src/{kevin,kernelbench,mirage,analysis,pipeline,utils}
mkdir -p kernels/{level_1,level_2,level_3,level_4}
mkdir -p data/{benchmarks,cache}
mkdir -p configs/hardware
mkdir -p experiments
mkdir -p notebooks
mkdir -p tests/fixtures/sample_kernels
mkdir -p scripts
mkdir -p reports/figures
mkdir -p docs

# Create root level files
echo "Creating root level files..."
touch README.md
touch requirements.txt
touch setup.py
touch .gitignore
touch .env.example

# Create source files
echo "Creating source files..."

# Kevin module
touch src/__init__.py
touch src/kevin/__init__.py
touch src/kevin/inference.py
touch src/kevin/models.py
touch src/kevin/prompts.py

# KernelBench module
touch src/kernelbench/__init__.py
touch src/kernelbench/runner.py
touch src/kernelbench/dataset.py
touch src/kernelbench/problems.py

# Mirage module
touch src/mirage/__init__.py
touch src/mirage/preprocessor.py
touch src/mirage/ir_converter.py
touch src/mirage/optimizer.py

# Analysis module
touch src/analysis/__init__.py
touch src/analysis/database.py
touch src/analysis/patterns.py
touch src/analysis/visualizations.py

# Pipeline module
touch src/pipeline/__init__.py
touch src/pipeline/optimizer.py
touch src/pipeline/config.py

# Utils module
touch src/utils/__init__.py
touch src/utils/cuda_utils.py
touch src/utils/hardware.py
touch src/utils/logging.py

# Create config files
echo "Creating config files..."
touch configs/kevin_config.yaml
touch configs/benchmark_config.yaml
touch configs/hardware/h100.yaml
touch configs/hardware/a100.yaml
touch configs/hardware/l40s.yaml

# Create experiment files
echo "Creating experiment files..."
touch experiments/__init__.py
touch experiments/run_benchmarks.py
touch experiments/analyze_results.py
touch experiments/compare_models.py

# Create notebook files
echo "Creating notebook files..."
touch notebooks/exploration.ipynb
touch notebooks/pattern_analysis.ipynb
touch notebooks/visualization.ipynb

# Create test files
echo "Creating test files..."
touch tests/__init__.py
touch tests/test_kevin.py
touch tests/test_kernelbench.py
touch tests/test_mirage.py

# Create script files
echo "Creating script files..."
touch scripts/setup_environment.sh
touch scripts/download_models.py
touch scripts/run_single_problem.py
touch scripts/generate_report.py

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Create documentation files
echo "Creating documentation files..."
touch docs/installation.md
touch docs/usage.md
touch docs/api.md
touch docs/contributing.md

# Create sample kernel directories for each level
echo "Creating sample kernel directories..."
for level in {1..4}; do
    for problem in {1..5}; do
        mkdir -p "kernels/level_${level}/problem_${problem}"
        touch "kernels/level_${level}/problem_${problem}/.gitkeep"
    done
done

# Create a basic .gitignore
echo "Creating .gitignore content..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Data
*.db
data/cache/
kernels/**/*.cu
kernels/**/*.cuh
kernels/**/*.json

# Models
*.pt
*.pth
*.bin
*.safetensors

# Logs
logs/
*.log

# Reports
reports/*.json
reports/figures/*.png
reports/figures/*.pdf

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db

# CUDA
*.cubin
*.ptx
*.fatbin
EOF

# Create a basic README
echo "Creating README content..."
cat > README.md << 'EOF'
# Kevin-Mirage-Optimization

A comprehensive framework for optimizing GPU kernels using Kevin-32B and Mirage.

## Overview

This project integrates:
- **Kevin-32B**: Multi-turn RL model for CUDA kernel generation
- **KernelBench**: Benchmark suite for GPU kernel optimization
- **Mirage**: Superoptimization framework for deep learning kernels

## Directory Structure

```
kevin-mirage-optimization/
├── src/                    # Main source code
├── kernels/               # Generated kernels
├── data/                  # Databases and cache
├── configs/               # Configuration files
├── experiments/           # Experiment scripts
├── notebooks/             # Analysis notebooks
├── tests/                 # Test suite
├── scripts/              # Utility scripts
├── reports/              # Generated reports
└── docs/                 # Documentation
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run setup: `bash scripts/setup_environment.sh`
4. Download models: `python scripts/download_models.py`

## Usage

```bash
# Run a single problem
python scripts/run_single_problem.py --level 1 --problem 40

# Run full benchmark
python experiments/run_benchmarks.py

# Analyze results
python experiments/analyze_results.py
```

## License

MIT
EOF

# Create a basic requirements.txt
echo "Creating requirements.txt content..."
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
datasets>=2.0.0

# Data processing
numpy>=1.20.0
pandas>=1.5.0
sqlalchemy>=2.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.0.0

# Development
jupyter>=1.0.0
ipykernel>=6.0.0
notebook>=6.5.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
click>=8.1.0
rich>=13.0.0
python-dotenv>=1.0.0

# CUDA compilation
ninja>=1.11.0
pybind11>=2.10.0
EOF

# Create a basic setup.py
echo "Creating setup.py content..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kevin-mirage-optimization",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GPU kernel optimization using Kevin-32B and Mirage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kevin-mirage-optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kevin-optimize=experiments.run_benchmarks:main",
        ],
    },
)
EOF

# Create .env.example
echo "Creating .env.example content..."
cat > .env.example << 'EOF'
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
TOGETHER_API_KEY=your_together_key_here
HUGGINGFACE_TOKEN=your_hf_token_here

# Model Paths
KEVIN_MODEL_PATH=cognition-ai/Kevin-32B
KEVIN_MODEL_CACHE_DIR=./data/cache/models

# Mirage Configuration
MIRAGE_PATH=/path/to/mirage
MIRAGE_BACKEND=cuda

# Hardware Configuration
CUDA_VISIBLE_DEVICES=0
GPU_CLOCK_LOCK=1350
GPU_MEMORY_FRACTION=0.9

# Database
DATABASE_PATH=./data/kernel_results.db
DATABASE_BACKUP_PATH=./data/backups

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/optimization.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Experiment Configuration
MAX_REFINEMENT_STEPS=8
NUM_TRAJECTORIES=16
BEAM_SEARCH_WIDTH=4
TIMEOUT_SECONDS=300

# Paths
KERNELS_OUTPUT_DIR=./kernels
REPORTS_OUTPUT_DIR=./reports
BENCHMARK_CACHE_DIR=./data/cache/benchmarks
EOF

# Create a sample notebook
echo "Creating sample notebook..."
cat > notebooks/exploration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kevin-Mirage Optimization Exploration\n",
    "\n",
    "This notebook provides an interactive environment for exploring optimization results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "from analysis.database import KernelDatabase\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create basic test structure
echo "Creating basic test files..."
cat > tests/test_kevin.py << 'EOF'
"""Tests for Kevin inference module."""

import pytest
from kevin.inference import KevinInference


class TestKevinInference:
    """Test cases for KevinInference class."""
    
    def test_initialization(self):
        """Test Kevin model initialization."""
        # Add tests here
        pass
    
    def test_kernel_extraction(self):
        """Test CUDA kernel extraction."""
        # Add tests here
        pass
EOF

# Create a basic run script
echo "Creating run script..."
cat > scripts/run_single_problem.py << 'EOF'
#!/usr/bin/env python
"""Quick script to run optimization on a single problem."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import click
from pipeline.optimizer import OptimizationPipeline


@click.command()
@click.option('--level', type=int, required=True, help='Problem level (1-4)')
@click.option('--problem', type=int, required=True, help='Problem ID')
@click.option('--refinements', type=int, default=4, help='Number of refinement steps')
@click.option('--trajectories', type=int, default=4, help='Number of trajectories')
def main(level, problem, refinements, trajectories):
    """Run optimization on a single KernelBench problem."""
    print(f"Optimizing Level {level} Problem {problem}")
    print(f"Refinements: {refinements}, Trajectories: {trajectories}")
    
    # Initialize pipeline
    pipeline = OptimizationPipeline()
    
    # Run optimization
    result = pipeline.optimize_problem(level, problem)
    
    print(f"\nResults:")
    print(f"Speedup: {result.get('kevin_speedup', 0):.2f}x")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/run_single_problem.py

# Final message
echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "Directory tree:"
tree -L 3 -I '__pycache__|*.pyc|.git' 2>/dev/null || find . -type d -not -path '*/\.*' | sed -e "s/[^-][^\/]*\//  /g" -e "s/^//" -e "s/-/|/" | head -20

echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "4. pip install -r requirements.txt"
echo "5. pip install -e ."
echo "6. cp .env.example .env  # Then add your API keys"
echo ""
echo "Happy coding! 🚀"
