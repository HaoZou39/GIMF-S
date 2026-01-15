# Multi-Objective to Single-Objective Parameterization Guide

## Overview

The codebase has been parameterized to support both single-objective and multi-objective optimization. You can now easily switch between different numbers of objectives by changing a single parameter.

## Key Changes

### 1. Added `num_objectives` Parameter

All modules now accept a `num_objectives` parameter:
- `num_objectives=1`: Single-objective TSP
- `num_objectives=2`: Bi-objective TSP (original)
- `num_objectives=N`: N-objective TSP (extensible)

### 2. Modified Files

#### Configuration (`train_motsp_n20.py`)
- Added `num_objectives` to both `env_params` and `model_params`
- Added consistency checks

#### Problem Definition (`MOTSProblemDef.py`)
- `get_random_problems()` now generates `(batch, nodes, 2*num_objectives)` coordinates
- Added `augment_xy_data_by_8_fold()` for general N-objective augmentation
- Kept `augment_xy_data_by_64_fold_2obj()` for backward compatibility

#### Environment (`MOTSPEnv.py`)
- `__init__` now stores `num_objectives`
- `load_problems()` uses parameterized data generation
- `_get_travel_distance()` computes distance for each objective in a loop

#### Model (`MOTSPModel.py`)
- Encoder: `nn.Linear(2*num_objectives, embedding_dim)` for node embedding
- Decoder: Hypernetwork input dimension = `num_objectives`

#### Trainer (`MOTSPTrainer.py`)
- Preference vector sampling:
  - Single objective: `pref = [1.0]`
  - Multi-objective: `pref = Dirichlet(alpha, ..., alpha)`
- Score extraction generalized to any number of objectives

## Usage

### Single-Objective TSP

```python
# In train_motsp_n20.py or create train_motsp_n20_single_obj.py
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 1,  # Single objective
}

model_params = {
    'embedding_dim': 128,
    'num_objectives': 1,  # Must match env_params
    'in_channels': 1,     # One channel for image
    # ... other params
}
```

### Bi-Objective TSP (Original)

```python
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 2,  # Bi-objective
}

model_params = {
    'embedding_dim': 128,
    'num_objectives': 2,
    'in_channels': 2,
    # ... other params
}
```

### Tri-Objective TSP

```python
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 3,  # Tri-objective
}

model_params = {
    'embedding_dim': 128,
    'num_objectives': 3,
    'in_channels': 3,
    # ... other params
}
```

## Training Flow

### Data Flow

```
Input: problems (batch, problem_size, 2*num_objectives)
  ↓
Environment: Creates image with num_objectives channels
  ↓
Encoder: 
  - Node embedding: Linear(2*num_objectives → 128)
  - Image embedding: patches (512 = 16*16*num_objectives → 128)
  ↓
Fusion Layers (objective-agnostic)
  ↓
Decoder:
  - Preference: [w1, w2, ..., wN] where sum=1
  - Hypernetwork: pref → decoder weights
  ↓
Output: Tour selection
  ↓
Reward: (batch, pomo, num_objectives)
  ↓
Scalarization: weighted_reward = (pref * reward).sum()
  ↓
Loss: Policy Gradient with POMO baseline
```

### Key Design Decisions

1. **Preference Vector**:
   - Single objective: `[1.0]` (no preference needed)
   - Multi-objective: Sampled from Dirichlet distribution

2. **Hypernetwork**:
   - Input dimension = `num_objectives`
   - Still generates 5 sets of decoder weights
   - For single objective, embedding dimension = max(2, num_objectives)

3. **Image Channels**:
   - `in_channels = num_objectives`
   - Each objective has its own channel in the image

4. **Distance Calculation**:
   - Loop over objectives
   - Each objective has independent 2D coordinates
   - Euclidean distance computed separately

## Backward Compatibility

✅ All original 2-objective code still works
✅ Default `num_objectives=2` if not specified
✅ Legacy functions (e.g., `augment_xy_data_by_64_fold_2obj`) preserved

## Testing

To test single-objective version:

```bash
cd GIMF-S/MOTSP/POMO
python train_motsp_n20_single_obj.py
```

To test with original 2-objective:

```bash
python train_motsp_n20.py
```

## Architecture Invariance

### Objective-Dependent Components
- ❌ Problem generation: `(batch, nodes, 2*num_objectives)`
- ❌ Image channels: `in_channels = num_objectives`
- ❌ Node embedding input: `Linear(2*num_objectives, 128)`
- ❌ Patch embedding input: `Linear(patch_size^2 * num_objectives, 128)`
- ❌ Distance calculation: loop over objectives
- ❌ Preference vector: dimension = `num_objectives`
- ❌ Reward: `(batch, pomo, num_objectives)`

### Objective-Agnostic Components
- ✅ Encoder layers (after embedding)
- ✅ Fusion layers
- ✅ Decoder architecture (after hypernetwork)
- ✅ POMO mechanism
- ✅ Policy gradient algorithm
- ✅ Baseline calculation

## Notes

1. **Data Augmentation**: 
   - 8-fold: Works for any number of objectives
   - 64-fold: Only for 2 objectives (original method)

2. **Training Efficiency**:
   - Single objective training is slightly faster (less distance computation)
   - Multi-objective training samples different preferences each batch

3. **Model Weights**:
   - Models trained with different `num_objectives` are **NOT** compatible
   - Need to retrain from scratch when changing objective count

