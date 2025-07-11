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
