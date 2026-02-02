# Decision Trees & Gradient Boosting Guide

A comprehensive, hands-on guide to understanding and mastering tree-based machine learning algorithms - from basic decision trees to state-of-the-art gradient boosting frameworks.

## Overview

This repository contains an interactive Jupyter notebook that takes you from foundational concepts to practical expertise with modern gradient boosting libraries. The guide emphasizes visual intuition, working code examples, and real-world best practices.

## What You'll Learn

### Part 1: Decision Tree Fundamentals
- The "20 Questions" mental model for understanding trees
- How splits work and why they matter
- Impurity metrics: Gini, Entropy, and MSE
- The bias-variance tradeoff in tree depth

### Part 2: Ensemble Methods
- Why single trees are unstable (high variance)
- Bagging and the "wisdom of crowds"
- Random Forests: combining bagging with feature randomization
- Boosting: learning from mistakes sequentially

### Part 3: XGBoost Deep Dive
- Second-order optimization and regularization
- Histogram-based split finding
- Built-in missing value handling
- Parameter tuning strategies

### Part 4: LightGBM
- Leaf-wise vs level-wise tree growth
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- The critical `num_leaves` parameter

### Part 5: CatBoost
- Ordered Target Statistics for categorical features
- Symmetric tree architecture
- Minimal hyperparameter tuning philosophy

### Part 6: Framework Comparison
- Head-to-head benchmarks on real data
- When to use each framework
- Sklearn's HistGradientBoosting as a baseline

### Part 7: Model Interpretation
- Built-in feature importance methods
- SHAP values for rigorous explanations
- Visualizing individual predictions

### Part 8: Best Practices
- Starting a new project workflow
- Learning rate and tree count coupling
- Reading learning curves
- When NOT to use tree-based models

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd DecisionTrees

# Create virtual environment and install dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook
```

### Dependencies

- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **matplotlib** - Plotting
- **seaborn** - Statistical visualization
- **scikit-learn** - ML algorithms and datasets
- **xgboost** - XGBoost framework
- **lightgbm** - LightGBM framework
- **catboost** - CatBoost framework
- **shap** - Model interpretability

## Usage

Open the notebook and run cells sequentially:

```bash
uv run jupyter notebook decision_trees_gradient_boosting_guide.ipynb
```

Each section builds on previous concepts. The notebook includes:
- Detailed explanations with visual diagrams
- Working code you can modify and experiment with
- Real datasets (California Housing, Breast Cancer)
- Performance comparisons and benchmarks

## Project Structure

```
DecisionTrees/
├── decision_trees_gradient_boosting_guide.ipynb  # Main tutorial notebook
├── LEARNING_GUIDE.md                              # Companion learning material
├── pyproject.toml                                 # Project dependencies
├── README.md                                      # This file
└── .venv/                                         # Virtual environment
```

## Prerequisites

- Basic Python programming
- Familiarity with NumPy and Pandas
- Understanding of basic ML concepts (train/test split, overfitting)

No prior knowledge of decision trees or gradient boosting is required.

## License

MIT
