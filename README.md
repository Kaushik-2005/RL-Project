# RL Bias Mitigation Project - Phase 1

**Group 3**: Aniketh (AM.EN.UA41E22009), Jatin (AM.EN.UA41E22024), Kaushik (AM.EN.UA41E22026)

## Project Overview

This project implements a Reinforcement Learning solution to mitigate gender bias in loan approval systems using Deep Q-Network (DQN) with fairness regularization.

### Problem Statement

Historical loan approval data shows significant gender bias:
- **Men approval rate**: 42.93%
- **Women approval rate**: 18.92%

The goal is to train an RL agent that learns fair loan approval decisions while maintaining accuracy.

## MDP Formulation

### State Space (S)
- `s_t = (salary_t, years_exp_t, sex_t)`
- Continuous + categorical features

### Action Space (A)
- `A = {0: Reject, 1: Approve}`
- Binary discrete action space

### Reward Function (R)
```
R = R_classification + R_fairness

R_classification:
  - Approve & label=Yes → +1
  - Approve & label=No → -1
  - Reject → 0

R_fairness = -λ * |approval_rate(women) - approval_rate(men)|
where λ = 0.5
```

### Discount Factor (γ)
- γ = 0.99

### Episode Structure
- 100 sequential applicants per episode
- Stochastic sampling from dataset

## Implementation

### Files Structure

```
.
├── biased_gender_loans.csv      # Dataset (10,000 loan applications)
├── loan_env.py                  # Custom Gym environment
├── dqn_agent.py                 # DQN agent with experience replay
├── train.py                     # Training utilities
├── visualization.py             # Plotting and analysis functions
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Key Components

1. **LoanApprovalEnv** (`loan_env.py`)
   - Custom OpenAI Gym environment
   - Implements MDP dynamics
   - Tracks fairness metrics (Statistical Parity Difference)

2. **DQNAgent** (`dqn_agent.py`)
   - Deep Q-Network with target network
   - Experience replay buffer
   - ε-greedy exploration strategy

3. **Training Pipeline** (`train.py`)
   - Episode-based training loop
   - Fairness metrics tracking
   - Model checkpointing

4. **Visualization** (`visualization.py`)
   - Training metrics plots
   - Fairness comparison charts
   - Dataset distribution analysis
   - Summary report generation

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```powershell
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:

```powershell
python main.py
```

This will:
1. Analyze baseline biased data
2. Train DQN agent (1000 episodes)
3. Evaluate trained agent
4. Generate visualizations and reports

### Output Files

- `dqn_loan_model.pt` - Trained model weights
- `training_metrics.png` - Training progress visualization
- `fairness_comparison.png` - Baseline vs RL agent comparison
- `data_distribution.png` - Dataset distribution analysis
- `summary_report.txt` - Detailed results summary

### Custom Training

```python
from train import train_dqn_agent
from loan_env import LoanApprovalEnv

# Train with custom parameters
agent, metrics, env = train_dqn_agent(
    data_path='biased_gender_loans.csv',
    num_episodes=2000,
    lambda_fairness=0.3,  # Adjust fairness weight
    learning_rate=0.0005,
    gamma=0.99
)
```

## Performance Metrics

The system tracks:

1. **Accuracy Metrics**
   - Total reward per episode
   - Classification accuracy

2. **Fairness Metrics**
   - Statistical Parity Difference (SPD): `P(approve|Woman) - P(approve|Man)`
   - Approval rates by gender
   - Disparity ratio

3. **Training Metrics**
   - Q-network loss
   - Epsilon decay
   - Buffer utilization

## Methodology

### DQN Algorithm

**Why DQN?**
- Handles discrete action spaces efficiently
- Supports complex non-linear reward structures
- Experience replay improves sample efficiency
- Target network stabilizes training

**Architecture**:
- Input: 3 features (salary, experience, gender)
- Hidden layers: [64, 64] with ReLU activation
- Output: Q-values for 2 actions

**Hyperparameters**:
- Learning rate: 0.001
- Batch size: 64
- Replay buffer: 10,000 experiences
- Target network update: every 10 episodes
- Exploration: ε from 1.0 → 0.01 (decay: 0.995)

### Fairness Regularization

The fairness penalty encourages demographic parity:
```
penalty = -0.5 * |approval_rate(women) - approval_rate(men)|
```

This is computed episodically and added to the classification reward.

## Expected Results

The trained agent should:
- ✓ Reduce Statistical Parity Difference by 50%+
- ✓ Maintain reasonable approval accuracy
- ✓ Balance fairness and performance
- ✓ Converge within 1000 episodes

## Limitations & Assumptions

1. **Historical labels as proxy**: Bank loan history used as ground truth
2. **Single fairness metric**: Only demographic parity enforced
3. **IID sampling**: Applicants sampled independently
4. **No repayment data**: Actual loan outcomes unknown

## Future Work (Phase 2)

- Implement policy gradient methods (PPO, A2C)
- Multi-objective optimization
- Additional fairness metrics (Equal Opportunity, Equalized Odds)
- Real-world dataset validation
- Counterfactual fairness analysis

## References

- Sutton & Barto - Reinforcement Learning: An Introduction
- Mnih et al. (2015) - Human-level control through deep RL
- Calders & Verwer (2010) - Three naive Bayes approaches for discrimination-free classification

## License

Academic project for educational purposes.

---

**Project Phase**: 1
**Course**: Reinforcement Learning
**Institution**: B.Tech SEM 7
**Date**: November 2025
