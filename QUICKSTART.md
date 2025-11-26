# Quick Start Guide - RL Bias Mitigation Project

## Installation

1. Install required packages:
```powershell
pip install numpy pandas matplotlib torch gym tqdm
```

Or use requirements file:
```powershell
pip install -r requirements.txt
```

## Running the Project

### Option 1: Run Complete Pipeline (Recommended)

Simply execute:
```powershell
python main.py
```

This will:
- ✓ Analyze baseline biased data
- ✓ Train DQN agent for 1000 episodes
- ✓ Evaluate trained agent
- ✓ Generate all visualizations and reports

Expected runtime: 5-15 minutes (depending on hardware)

### Option 2: Step-by-Step Execution

#### Step 1: Train the Agent
```python
from train import train_dqn_agent

agent, metrics, env = train_dqn_agent(
    data_path='biased_gender_loans.csv',
    num_episodes=1000,
    lambda_fairness=0.5
)
```

#### Step 2: Evaluate the Agent
```python
from train import evaluate_agent

eval_metrics = evaluate_agent(agent, env, num_episodes=100)
```

#### Step 3: Generate Visualizations
```python
from visualization import plot_training_metrics, plot_fairness_comparison, analyze_baseline_data

baseline_metrics = analyze_baseline_data('biased_gender_loans.csv')
plot_training_metrics(metrics)
plot_fairness_comparison(baseline_metrics, eval_metrics)
```

## Output Files

After running `main.py`, you will get:

1. **dqn_loan_model.pt** - Saved model weights
2. **training_metrics.png** - Training progress (4 subplots)
3. **fairness_comparison.png** - Baseline vs RL agent
4. **data_distribution.png** - Dataset analysis
5. **summary_report.txt** - Detailed text report

## Interpreting Results

### Key Metrics to Watch:

1. **Statistical Parity Difference (SPD)**
   - Measures: approval_rate(women) - approval_rate(men)
   - Goal: Close to 0 (perfect parity)
   - Baseline: ~-0.24 (women disadvantaged)

2. **Approval Rates**
   - Baseline: Men 42.93%, Women 18.92%
   - Target: Reduce the gap while maintaining accuracy

3. **Total Reward**
   - Should increase over episodes
   - Balances classification accuracy + fairness

## Troubleshooting

### Issue: Import errors
**Solution**: Install missing packages
```powershell
pip install <package_name>
```

### Issue: CUDA errors (if using GPU)
**Solution**: The code automatically uses CPU. For GPU:
```python
# In dqn_agent.py, add:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Issue: Training takes too long
**Solution**: Reduce episodes in main.py:
```python
NUM_EPISODES = 500  # Instead of 1000
```

### Issue: Memory errors
**Solution**: Reduce batch size or buffer size in main.py:
```python
BATCH_SIZE = 32  # Instead of 64
```

## Customization

### Adjust Fairness Weight

In `main.py`, modify:
```python
LAMBDA_FAIRNESS = 0.3  # Lower = more focus on accuracy
LAMBDA_FAIRNESS = 0.7  # Higher = more focus on fairness
```

### Change Network Architecture

In `dqn_agent.py`, modify `QNetwork`:
```python
hidden_dims = [128, 128]  # Larger network
```

### Modify Training Duration

In `main.py`:
```python
NUM_EPISODES = 2000  # More training
MAX_STEPS = 50000    # More total steps
```

## Expected Results

After training, you should see:

✓ SPD reduction by 40-60%
✓ Approval rates becoming more balanced
✓ Reward increasing and stabilizing
✓ Loss decreasing over time

## Next Steps

1. Experiment with different λ values
2. Try different network architectures
3. Implement additional fairness metrics
4. Test on different datasets

## Support

For questions or issues:
- Check README.md for detailed documentation
- Review the code comments
- Verify all dependencies are installed

---

**Happy Learning!** 🚀
