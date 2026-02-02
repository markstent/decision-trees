# Decision Trees & Gradient Boosting - Learning Guide

A companion reference for the interactive notebook. Use this document for quick lookups, concept review, and as a study guide.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Decision Trees](#2-decision-trees)
3. [Ensemble Methods](#3-ensemble-methods)
4. [Gradient Boosting Frameworks](#4-gradient-boosting-frameworks)
5. [Hyperparameter Reference](#5-hyperparameter-reference)
6. [Model Interpretation](#6-model-interpretation)
7. [Practical Guidelines](#7-practical-guidelines)
8. [Quick Reference Tables](#8-quick-reference-tables)

---

## 1. Core Concepts

### The Machine Learning Pipeline for Trees

```
Data -> Feature Engineering -> Train/Val/Test Split -> Model Training -> Evaluation -> Interpretation
```

### Key Terminology

| Term | Definition |
|------|------------|
| **Node** | A decision point in the tree |
| **Root** | The topmost node (first split) |
| **Leaf** | Terminal node containing a prediction |
| **Depth** | Maximum distance from root to any leaf |
| **Split** | Division of data based on a feature threshold |
| **Impurity** | Measure of class mixture in a node |
| **Ensemble** | Combination of multiple models |
| **Boosting** | Sequential ensemble where each model corrects previous errors |
| **Bagging** | Parallel ensemble trained on bootstrap samples |

---

## 2. Decision Trees

### How Trees Make Decisions

A decision tree recursively partitions the feature space by asking binary questions:

```
Is feature X > threshold?
├── Yes -> Go right
└── No  -> Go left
```

This continues until reaching a leaf node, which contains the prediction.

### Impurity Metrics

#### Gini Impurity (Classification)

Measures the probability of misclassifying a randomly chosen element:

```
Gini = 1 - sum(p_i^2)
```

- **Range**: 0 (pure) to 0.5 (maximum impurity for binary)
- **Interpretation**: Lower is better
- **Default** in scikit-learn

#### Entropy / Information Gain (Classification)

Measures the average information needed to classify an element:

```
Entropy = -sum(p_i * log2(p_i))
```

- **Range**: 0 (pure) to 1 (maximum uncertainty for binary)
- **Information Gain** = Parent Entropy - Weighted Child Entropy

#### Mean Squared Error (Regression)

```
MSE = (1/n) * sum((y_i - y_mean)^2)
```

The best split minimizes the weighted average MSE of child nodes.

### The Bias-Variance Tradeoff

| Tree Depth | Bias | Variance | Risk |
|------------|------|----------|------|
| Shallow (2-4) | High | Low | Underfitting |
| Medium (5-10) | Balanced | Balanced | Often optimal |
| Deep (15+) | Low | High | Overfitting |

**Key insight**: A single decision tree has high variance - small changes in training data can produce very different trees.

---

## 3. Ensemble Methods

### Bagging (Bootstrap Aggregating)

**Goal**: Reduce variance by averaging multiple high-variance models.

**Process**:
1. Create N bootstrap samples (random sampling with replacement)
2. Train one tree on each sample
3. Average predictions (regression) or vote (classification)

**Why it works**: Averaging reduces variance by a factor of ~1/N while keeping bias constant.

### Random Forest

**Innovation**: Bagging + random feature selection at each split.

**Key parameters**:
- `n_estimators`: Number of trees (more is generally better, diminishing returns)
- `max_features`: Features considered at each split (sqrt(n) for classification, n/3 for regression)
- `max_depth`: Tree depth limit

**Strengths**:
- Robust to overfitting
- Handles high-dimensional data well
- Provides feature importance
- Embarrassingly parallel

### Boosting

**Goal**: Reduce bias by sequentially correcting errors.

**Process**:
1. Train a weak learner on the data
2. Calculate residuals (errors)
3. Train next learner to predict the residuals
4. Combine predictions: `F(x) = f1(x) + f2(x) + ... + fn(x)`

**Key insight**: Each tree learns what previous trees got wrong.

### Gradient Boosting

Generalizes boosting by using gradient descent in function space:

```
F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
```

Where `h_m` is fitted to the negative gradient of the loss function.

---

## 4. Gradient Boosting Frameworks

### XGBoost

**Key innovations**:
1. **Regularized objective**: Adds L1/L2 penalties to prevent overfitting
2. **Second-order optimization**: Uses both gradient and Hessian for better convergence
3. **Histogram binning**: Discretizes features for faster split finding
4. **Missing value handling**: Learns optimal direction for missing values
5. **Column subsampling**: Reduces correlation between trees

**Best for**:
- Structured/tabular data competitions
- When you need maximum predictive performance
- Datasets with missing values

### LightGBM

**Key innovations**:
1. **Leaf-wise growth**: Grows the leaf with highest gain (vs level-wise)
2. **GOSS**: Keeps all large-gradient instances, samples small-gradient ones
3. **EFB**: Bundles mutually exclusive features together
4. **Histogram-based**: Uses 255 bins by default

**Best for**:
- Large datasets (millions of rows)
- High-dimensional data
- When training speed matters

**Caution**: Leaf-wise growth can overfit on small datasets. Use `num_leaves` carefully.

### CatBoost

**Key innovations**:
1. **Ordered Target Statistics**: Handles categoricals without target leakage
2. **Symmetric trees**: All nodes at same depth use same split (faster inference)
3. **Ordered boosting**: Prevents target leakage during training

**Best for**:
- Datasets with many categorical features
- When you want minimal hyperparameter tuning
- Production systems (fast inference)

### Sklearn HistGradientBoosting

**Characteristics**:
- Native histogram-based implementation
- Good baseline with minimal dependencies
- Handles missing values natively

**Best for**:
- Quick prototyping
- When you want to avoid extra dependencies
- Educational purposes

---

## 5. Hyperparameter Reference

### Universal Parameters

| Parameter | XGBoost | LightGBM | CatBoost | Effect |
|-----------|---------|----------|----------|--------|
| Learning rate | `learning_rate` | `learning_rate` | `learning_rate` | Lower = more trees needed, better generalization |
| Number of trees | `n_estimators` | `n_estimators` | `iterations` | More = better (with early stopping) |
| Tree depth | `max_depth` | `max_depth` | `depth` | Deeper = more complex patterns |
| Min samples leaf | `min_child_weight` | `min_data_in_leaf` | `min_data_in_leaf` | Higher = more regularization |

### Framework-Specific Critical Parameters

#### XGBoost
```python
params = {
    'learning_rate': 0.1,      # Start here, reduce for final model
    'max_depth': 6,            # 3-10 typical range
    'min_child_weight': 1,     # Increase for noisy data
    'subsample': 0.8,          # Row sampling
    'colsample_bytree': 0.8,   # Feature sampling
    'reg_alpha': 0,            # L1 regularization
    'reg_lambda': 1,           # L2 regularization
}
```

#### LightGBM
```python
params = {
    'learning_rate': 0.1,
    'num_leaves': 31,          # KEY PARAMETER: 2^max_depth - 1 equivalent
    'max_depth': -1,           # Usually leave unlimited, control via num_leaves
    'min_data_in_leaf': 20,    # Increase for small datasets
    'feature_fraction': 0.8,   # Column sampling
    'bagging_fraction': 0.8,   # Row sampling
    'bagging_freq': 1,         # Enable bagging
}
```

#### CatBoost
```python
params = {
    'learning_rate': 0.1,
    'depth': 6,                # Symmetric tree depth
    'l2_leaf_reg': 3,          # L2 regularization
    'random_strength': 1,      # Randomization for scoring splits
    'bagging_temperature': 1,  # Controls bootstrap sampling
}
```

### The Learning Rate / Trees Tradeoff

```
learning_rate=0.3, n_estimators=100  ->  Fast training, possibly underfitting
learning_rate=0.1, n_estimators=300  ->  Good balance
learning_rate=0.01, n_estimators=3000 -> Slow training, often best results
```

**Rule of thumb**: When you decrease learning rate by 10x, increase trees by 10x.

---

## 6. Model Interpretation

### Feature Importance Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Gain** | Total gain from splits using feature | Considers magnitude | Biased toward high-cardinality |
| **Split count** | Number of times feature is used | Simple | Ignores split quality |
| **Permutation** | Accuracy drop when feature is shuffled | Model-agnostic | Slow, affected by correlation |
| **SHAP** | Game-theoretic attribution | Theoretically grounded, local | Computationally expensive |

### SHAP Values

SHAP (SHapley Additive exPlanations) provides:
- **Global importance**: Average absolute SHAP value per feature
- **Local explanations**: Contribution of each feature to individual predictions
- **Interaction effects**: How features combine

**Key plots**:
- **Summary plot**: Feature importance + distribution of effects
- **Waterfall plot**: Breakdown of single prediction
- **Dependence plot**: Feature value vs SHAP value

---

## 7. Practical Guidelines

### Starting a New Project

1. **Baseline**: Train sklearn's `HistGradientBoostingClassifier/Regressor` with defaults
2. **Quick win**: Try LightGBM with defaults (often faster)
3. **Tune**: If needed, tune `learning_rate`, `num_leaves`/`max_depth`, regularization
4. **Final model**: Lower learning rate, more trees, early stopping

### Reading Learning Curves

| Pattern | Diagnosis | Solution |
|---------|-----------|----------|
| Train/Val both high error | Underfitting | Increase complexity, more features |
| Train low, Val high | Overfitting | More regularization, less depth |
| Val error still decreasing | More trees needed | Increase n_estimators |
| Val error increasing | Too many trees | Use early stopping |

### When NOT to Use Tree-Based Models

- **Image/audio/text**: Use deep learning
- **Very small datasets** (<100 samples): Simple models may generalize better
- **Need smooth decision boundaries**: Trees create axis-aligned rectangles
- **Extrapolation required**: Trees cannot predict outside training range
- **Real-time inference constraints**: Single trees are fast, ensembles less so

### Common Pitfalls

1. **Target leakage**: Feature that contains information about the target
2. **Time series without proper splits**: Use time-based validation
3. **Ignoring class imbalance**: Use `scale_pos_weight` or sampling
4. **Over-tuning**: Use cross-validation, not single validation set
5. **Feature engineering neglect**: Trees still benefit from good features

---

## 8. Quick Reference Tables

### Framework Comparison

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Speed | Fast | Fastest | Moderate |
| Memory | Moderate | Low | Higher |
| Categorical handling | Manual encoding | Native (limited) | Best native |
| Missing values | Native | Native | Native |
| GPU support | Yes | Yes | Yes |
| Default performance | Good | Good | Often best OOB |
| Tuning required | Medium | Medium | Low |

### Default Parameter Recipes

#### "Quick and Good" (Start here)
```python
# Works for most problems
params = {
    'learning_rate': 0.1,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
}
```

#### "Maximum Performance" (Competition)
```python
params = {
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'early_stopping_rounds': 100,
    'max_depth': 7,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
}
```

#### "Fast Prototyping"
```python
params = {
    'learning_rate': 0.3,
    'n_estimators': 100,
    'max_depth': 4,
}
```

#### "Small Dataset" (<1000 samples)
```python
params = {
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': 3,
    'min_child_weight': 5,  # XGBoost
    'num_leaves': 7,        # LightGBM
}
```

### Metric Reference

| Task | Metric | When to Use |
|------|--------|-------------|
| Classification | Accuracy | Balanced classes |
| Classification | AUC-ROC | Ranking matters |
| Classification | F1 | Imbalanced classes |
| Classification | Log Loss | Probability calibration |
| Regression | RMSE | Penalize large errors |
| Regression | MAE | Robust to outliers |
| Regression | R-squared | Explained variance |

---

## Further Reading

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Original XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [CatBoost Paper](https://arxiv.org/abs/1706.09516)
