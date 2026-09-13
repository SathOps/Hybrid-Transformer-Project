# Binary Particle Swarm Optimization (BPSO) In-Depth Analysis

## 1. BPSO Hyperparameters & Search Strategy

- **Swarm Size**: 20 particles
- **Maximum Iterations**: 20 iterations
- **Inertia Weight ($w$)**: 0.7
- **Cognitive Coefficient ($c1$)**: 1.5
- **Social Coefficient ($c2$)**: 1.5
- **Velocity Bounds ($v_{min}, v_{max}$)**: [-6.0, 6.0]
- **Transfer Function**: Sigmoid probability $S(v) = rac{1}{1 + \exp(-v)}$
- **Random Seed**: 42
- **Fitness Evaluator**: Lightweight XGBoost (`hist`, `n_estimators=30`, `max_depth=5`)
- **Fitness Objective**: **Validation Macro F1-Score** on fixed 200,000 train / 200,000 val subset.

---

## 2. Optimization Convergence Trajectory

- **Iteration 1**: Fitness (Val Macro F1) = `0.650834` (30 features selected)
- **Iteration 2**: Fitness = `0.658467` (28 features selected)
- **Iteration 5**: Fitness = `0.659006` (26 features selected)
- **Iteration 7**: Fitness = `0.660370` (23 features selected)
- **Iteration 8**: Fitness = `0.661252` (26 features selected)
- **Iteration 12**: Fitness = `0.662544` (29 features selected)
- **Iteration 13 (Global Best)**: Fitness = **`0.662942`** (27 features selected)
- **Iterations 14–20**: Search converged; no higher macro F1 found.

Total BPSO search execution duration: **243.34 seconds** (~4.0 minutes).

---

## 3. Model-Dependent BPSO Impact Analysis

> [!IMPORTANT]
> **Scientific Finding**: BPSO feature selection does **NOT** universally improve all classifier architectures. Its efficacy depends heavily on the model's inductive bias:

1. **Tree-Based Models (XGBoost)**:
   - XGBoost naturally constructs axis-aligned splits. Redundant features increase tree search space.
   - BPSO pruned 19 uninformative features (41.3%), reducing XGBoost training time from ~75s to 44.58s (**-40.6% training time**) with only a nominal -0.35% drop in test accuracy (`74.91%` → `74.56%`).

2. **Sequential Neural Models (CNN-Transformer)**:
   - 1D-CNN filters rely on contiguous spatial correlations across adjacent feature channels.
   - Removing 19 feature columns altered the feature sequence topography, causing overall test accuracy to decline from `56.40%` to `53.80%`.
