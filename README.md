# CSC311 Project: Collaborative Filtering and Item Response Theory

**Authors:** Pete Chen, Ocean Chen, Harvi Karatha  
**Date:** April 2024

## Project Overview

This project explores educational data modeling through collaborative filtering and Item Response Theory (IRT) models to predict student performance on diagnostic questions. The work is divided into two main parts: implementing baseline collaborative filtering methods and developing enhanced IRT models.

## Part A: Collaborative Filtering and Baseline Models

### 1. k-Nearest Neighbors (kNN) Collaborative Filtering

#### User-Based Collaborative Filtering
- **Best k:** 11
- **Test Accuracy:** 0.6842
- Assumes students with similar response patterns will perform similarly on new questions

#### Item-Based Collaborative Filtering  
- **Best k:** 21
- **Test Accuracy:** 0.6816
- Assumes questions with similar student response patterns are of similar difficulty

#### Performance Comparison
User-based CF slightly outperformed item-based CF by ~0.003 accuracy points.

#### kNN Limitations
1. **Time Complexity:** O(nd) where n = data points, d = features
2. **Cold Start Problem:** Poor performance with new students/questions
3. **Curse of Dimensionality:** Struggles with high-dimensional sparse data

### 2. One-Parameter Item Response Theory (1PL)

#### Mathematical Foundation
- **Probability Formula:** `P(c_ij = 1) = exp(θ_i - β_j) / (1 + exp(θ_i - β_j))`
- **Parameters:** 
  - θ_i: Student i's ability
  - β_j: Question j's difficulty

#### Performance
- **Optimal Hyperparameters:** Learning rate = 0.018, Iterations = 30
- **Test Accuracy:** 0.7076
- **Validation Accuracy:** 0.7070

### 3. Neural Network Matrix Factorization

#### Implementation Details
- **Architecture:** Embedding layers for users and items with dot product interaction
- **Best Parameters:** k=10, learning rate=0.01, epochs=100
- **Performance:** 
  - Test Accuracy: 0.6924
  - Validation Accuracy: 0.6860 (with L2 regularization λ=0.001: 0.6877)

## Part B: Enhanced IRT Models

### Two-Parameter Logistic (2PL) Model

#### Enhanced Formula
```
P(c_ij = 1) = exp(α_j(θ_i - β_j)) / (1 + exp(α_j(θ_i - β_j)))
```

**New Parameter:**
- α_j: Discrimination parameter (how well question j differentiates student abilities)

#### Performance
- **Test Accuracy:** 0.7056
- **Validation Accuracy:** 0.7083
- **Best performer among all models**

### Three-Parameter Logistic (3PL) Model

#### Complete Formula
```
P(c_ij = 1) = γ_j + (1 - γ_j) * exp(α_j(θ_i - β_j)) / (1 + exp(α_j(θ_i - β_j)))
```

**Additional Parameter:**
- γ_j: Guessing parameter (probability of correct answer by chance)

#### Key Finding: Guessing Parameter Irrelevance
- **Average γ:** -0.003196 (converges to ~0)
- **Average α:** 1.206
- **Performance:** Slightly worse than 2PL
- **Conclusion:** Guessing has negligible impact on diagnostic questions

### Model Comparison Summary

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| User-based kNN | 0.6922 | 0.6816 |
| Neural Network | 0.6877 | 0.6833 |
| 1PL IRT | 0.7070 | 0.7076 |
| **2PL IRT** | **0.7083** | **0.7056** |
| 3PL IRT | 0.7079 | 0.7036 |

## Key Insights

### Why 2PL Outperforms Other Models
1. **Discrimination Parameter Importance:** Questions vary significantly in their ability to differentiate students
2. **Diagnostic Question Nature:** Dataset consists of misconception-focused questions where discrimination matters more than guessing
3. **Optimal Complexity:** Adds meaningful parameter (α) without overfitting (unlike 3PL with γ)

### Why Guessing Parameter (γ) Fails
1. **Question Design:** Diagnostic questions designed to capture specific misconceptions, not random guessing
2. **Convergence to Zero:** γ consistently approaches 0 regardless of initialization
3. **Overfitting:** Additional complexity without explanatory power

## Technical Implementation

### Mathematical Derivations
- Complete log-likelihood derivations for all IRT models
- Gradient computations for optimization
- Proper handling of sigmoid activation functions

### Optimization Strategies
- Grid search for hyperparameter tuning
- Learning rate scheduling
- L2 regularization for neural networks

## Limitations and Future Work

### Current Limitations
1. **Parameter Correlation:** Potential covariance between discrimination (α) and difficulty (β) parameters
2. **Dataset Specificity:** Models tuned for diagnostic questions may not generalize
3. **Computational Complexity:** IRT models require iterative optimization


## Results

The project demonstrates that **Two-Parameter Logistic (2PL) IRT** provides the best performance for educational diagnostic data, achieving 70.83% validation accuracy. Key findings include:

- Discrimination parameter (α) is crucial for modeling question-specific difficulty variations
- Guessing parameter (γ) shows minimal impact in diagnostic contexts
- IRT models significantly outperform traditional collaborative filtering approaches
- Neural network matrix factorization provides competitive but inferior performance to IRT models

## Conclusion

This work provides a solid foundation for educational assessment modeling and highlights the importance of parameter selection in IRT applications. The 2PL model strikes the optimal balance between model complexity and predictive performance for diagnostic educational data.

## References

1. Thompson, Nathan, PhD. "What Is the Three Parameter IRT Model (3PL)?" Assessment Systems, 28 Feb. 2024
2. "Item Response Theory." Wikipedia, 18 Feb. 2024
