# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Create the conda environment from the provided environment file:
```bash
conda env create -f CFA_environment.yml
conda activate CFA
```

**Key Dependencies:**
- Python 3.8.8
- PyTorch 1.10.1 with CUDA 11.3
- PyTorch Geometric 2.0.3
- scikit-learn, numpy, pandas, scipy

## Project Architecture

This is a fairness-aware machine learning research project implementing the Comprehensive Fairness Algorithm (CFA) for fair model explanations.

**Core Components:**
- `mlp.py`: Multi-layer perceptron model with hidden representation extraction
- `train.py`: Main training script with fairness-aware loss functions
- `test.py`: Testing script for evaluating trained models
- `utils.py`: Data loading utilities for different datasets (german, bail, math, por)
- `explanation_metrics.py`: GraphLIME-based explanation fairness evaluation
- `parse.py`: Command-line argument parsing

**Training Architecture:**
- Warm-start training (first epochs): Standard classification loss only
- Fair training (after `opt_start_epoch`): Classification + distance-based fairness loss
- Distance loss computed on both original and masked features (via GraphLIME)
- Loss: `l_classification + λ*(l_distance + l_distance_masked)`

## Common Commands

**Training Models:**
```bash
# Train on German dataset with hyperparameter search
bash run_german.sh

# Train single model with specific parameters
python train.py --epochs 500 --dataset german --lr 1e-3 --lambda_ 1.0 --seed 1

# Train with similarity-based data splitting
python train.py --epoch 450 --use_similarity_split

# Advanced training variants
python train_adv.py  # Adversarial training
python train_monitor.py  # Training with monitoring
```

**Testing Models:**
```bash
# Test all datasets with multiple seeds
bash run_best.sh

# Test specific dataset
python test.py --dataset german --seed 1
```

**Data Analysis:**
```bash
# Test similarity-based splitting
python test_similarity_split.py
```

## Datasets

Four datasets are supported: `german`, `bail`, `math`, `por`
- Each dataset has a CSV file with features and a corresponding edges file for graph structure
- Data loading handled automatically through `load_data_util()` in `utils.py`
- Each dataset has sensitive attributes for fairness evaluation

## Key Parameters

**Training Parameters:**
- `--lambda_`: Coefficient for fairness loss (0 to 10, commonly 0.001-1)
- `--epochs`: Training epochs (typically 500-1000)
- `--opt_start_epoch`: When to start applying fairness loss (default varies)
- `--lr`: Learning rate (commonly 1e-2 to 1e-4)
- `--dropout`: Dropout rate (0.1, 0.3, 0.5)
- `--weight_decay`: L2 regularization (1e-3 to 1e-5)

**Explanation Parameters:**
- `--topK`: Number of features to mask for fidelity computation
- `--top_ratio`: Ratio of top features for explanation

## Evaluation Metrics

The framework evaluates models on:
1. **Utility**: AUC, F1-score, Accuracy
2. **Traditional Fairness**: Demographic Parity (DP), Equal Opportunity (EO)
3. **Explanation Fairness**: VEF (Variance in Explanation Fidelity), REF (Relative Explanation Fidelity)
4. **Comprehensive Score**: `(auc+f1+acc)/3 - (dp+eo)/2 - (vef+ref)/2 + log(I)`

## Workflow

1. **Hyperparameter Search**: Use shell scripts like `run_german.sh` to search optimal parameters
2. **Best Parameter Selection**: Use `scripts/Best Validation.ipynb` to analyze validation logs
3. **Final Testing**: Run `test.py` with optimal parameters using `run_best.sh`
4. **Result Analysis**: Use `scripts/Test Result.ipynb` to analyze final results

## File Organization

- `dataset/`: Contains CSV data files and edge files for graph structure
- `scripts/`: Jupyter notebooks for result analysis
- `weights/`: Saved model checkpoints (created during training)
- `train_logs/`: Training logs and validation records
- `PraFFL/`, `Procedural-Fairness-Relationship/`: Additional research modules

## Development Notes

- The model uses GraphLIME for generating explanations and masked features
- Distance-based fairness loss operates on hidden representations
- Supports both standard and adversarial training modes
- All models are saved automatically during training for later evaluation

Develop goal for next step:
1. create a new train file, refer train.py for basic struct and format

Method 1: Attention-Based Fairness
Goal: Reduce bias by ensuring that attention weights for sensitive groups (e.g., male vs female) are similar.

Process:
Extract attention weights from the model using interpretability tools such as SHAP values or Integrated Gradients.(please provide tools like SHAP, IG, LIME. then we will decide which to use after experiment)
Split data into sensitive groups (for example, male and female).
Compute separate attention vectors for each group.
Compare these vectors by measuring the divergence between them. Jensen-Shannon Divergence (JSD) is used as the metric to quantify the difference.
Fairness integration:
Add the divergence score as a penalty term in the training loss. (like the lambda term)
This encourages the model to learn representations where the influence of the sensitive label (e.g., sex) is minimized.
Final training loss = original utility loss + fairness penalty (from divergence).

After designing the algrthim, try to run 500/1000 epoch for a little experiment, use bail dataset as default

The current train_attention_fairness.py has a few issues
1. we have a few extract weight method like shap,ig,lime,gradient choices=['shap', 'integrated_gradients', 'lime', 'gradient_shap'],
not all of them work, some of them can't train due to logic bug, some of them can run but the result is weirld.

If you want to run, please run this command: python train_attention_fairness.py --dataset por --epochs 2000 --attention_method integrated_gradients --use_attention_fairness
It will print out some training result

2. the attention term didn't decrease with the training process, we need to figure out what's going on?

four methods: 
shap : too time comsuming
intergrated_gradients: good
lime: maybe too slow
gradient_shap

(dataset four options)
(epoch 2000)
(attention method: either intergrated_gradients or gradient_shap)
(lambda: either 1,5,10)
record 8 values for testing: ACC,F1,AUC,DP,EO,REF,VEF,JSD
python train_attention_fairness.py --dataset por --epochs 2000 --attention_method integrated_gradients --use_attention_fairness