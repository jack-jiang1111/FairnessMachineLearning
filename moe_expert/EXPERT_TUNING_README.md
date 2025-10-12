# Expert Hyperparameter Tuning

This directory contains scripts for tuning the hyperparameters of individual experts in the Mixture of Experts (MoE) system, following Stage 1 of the training pipeline described in the main README.

## Overview

The expert tuning process implements the first stage of the MoE training pipeline:

1. **Expert 1 (Baseline)**: Tuned for utility score (accuracy + f1 + auc) / 3
2. **Expert 2 (Result Fairness)**: Tuned for fairness result (DP + EO) / 2  
3. **Expert 3 (Procedural Fairness)**: Tuned for fairness procedure (REF + VEF + ATT_JSD) / 3

## Files

- `tune_experts.py`: Main hyperparameter tuning implementation
- `run_expert_tuning.py`: Example script to run expert tuning
- `EXPERT_TUNING_README.md`: This documentation

## Usage

### Basic Usage

```bash
# Tune experts for bail dataset with 30 trials per expert
python moe_expert/run_expert_tuning.py --dataset bail --n_trials 30

# Tune experts for german dataset with 50 trials per expert
python moe_expert/run_expert_tuning.py --dataset german --n_trials 50
```

### Advanced Usage

```python
from moe_expert.tune_experts import ExpertTuner

# Create tuner
tuner = ExpertTuner(
    dataset='bail',
    seed=0,
    cache_dir='weights/moe_experts'
)

# Tune all experts
results = tuner.tune_all_experts(n_trials=50)

# Or tune individual expert
expert1_results = tuner.tune_expert('expert1', n_trials=30)
```

## Hyperparameter Search Spaces

### Expert 1 (Baseline)
- `lr`: [1e-5, 1e-3] (log uniform)
- `weight_decay`: [1e-6, 1e-3] (log uniform)

### Expert 2 (Result Fairness)
- `lr`: [1e-5, 1e-3] (log uniform)
- `weight_decay`: [1e-6, 1e-3] (log uniform)
- `lambda_rep`: [0.01, 10.0] (log uniform)
- `lambda_fair`: [0.01, 10.0] (log uniform)

### Expert 3 (Procedural Fairness)
- `lr`: [1e-5, 1e-3] (log uniform)
- `weight_decay`: [1e-6, 1e-3] (log uniform)
- `lambda_attention`: [0.01, 10.0] (log uniform)
- `lambda_adv`: [0.01, 10.0] (log uniform)

## Scoring Metrics

Each expert is optimized for its specialization:

- **Expert 1**: Maximizes utility score = (accuracy + f1 + auc) / 3
- **Expert 2**: Minimizes fairness result = (DP + EO) / 2
- **Expert 3**: Minimizes fairness procedure = (REF + VEF + ATT_JSD) / 3

## Output Files

The tuning process creates several output files in the cache directory:

### Model Checkpoints
- `{dataset}_expert1_best.pt`: Best Expert 1 weights
- `{dataset}_expert2_best.pt`: Best Expert 2 weights  
- `{dataset}_expert3_best.pt`: Best Expert 3 weights

### Tuning Results
- `{dataset}_expert1_tuning_results.json`: Detailed Expert 1 tuning results
- `{dataset}_expert2_tuning_results.json`: Detailed Expert 2 tuning results
- `{dataset}_expert3_tuning_results.json`: Detailed Expert 3 tuning results
- `{dataset}_expert_tuning_summary.json`: Overall tuning summary

## Example Results

```json
{
  "expert_type": "expert1",
  "best_params": {
    "lr": 0.000234,
    "weight_decay": 0.0000123
  },
  "best_score": 0.8456,
  "best_metrics": {
    "acc": 0.8234,
    "f1": 0.8123,
    "auc": 0.9011,
    "utility": 0.8456
  }
}
```

## Integration with Gate Training

After expert tuning is complete, the best expert configurations can be used in gate training:

```python
# Load tuned experts for gate training
trainer = MoETrainer(
    dataset='bail',
    use_cached_experts=True,  # Load tuned experts
    cache_dir='weights/moe_experts'
)
```

## Tips for Effective Tuning

1. **Number of Trials**: Use 30-50 trials per expert for good results
2. **Early Stopping**: The tuning uses early stopping to prevent overfitting
3. **Validation**: Each expert is evaluated on validation data during tuning
4. **Model Selection**: Best models are selected based on expert-specific metrics
5. **Caching**: Results are automatically saved for later use

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Convergence Issues**: Increase number of trials or adjust search space
3. **Slow Training**: Use fewer trials or reduce epochs per trial

### Performance Tips

1. Use GPU if available for faster training
2. Start with fewer trials to test the setup
3. Monitor validation metrics during tuning
4. Save intermediate results regularly

## Next Steps

After expert tuning is complete:

1. Use the tuned experts for gate training (Stage 2)
2. Evaluate the full MoE system on test data
3. Optionally run fine-tuning (Stage 3) for final integration
