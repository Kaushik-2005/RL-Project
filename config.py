"""
Configuration file for RL Bias Mitigation Project
Modify these parameters to experiment with different settings
"""

# Data Configuration
DATA_PATH = 'biased_gender_loans.csv'
MODEL_SAVE_PATH = 'dqn_loan_model.pt'

# Environment Configuration
EPISODE_LENGTH = 100  # Number of applicants per episode
LAMBDA_FAIRNESS = 0.5  # Fairness penalty weight (0.0 = no fairness, 1.0 = max fairness)

# Training Configuration
NUM_EPISODES = 1000  # Total number of training episodes
MAX_STEPS = 30000    # Maximum total training steps
PRINT_FREQ = 50      # Print progress every N episodes

# DQN Hyperparameters
LEARNING_RATE = 0.001       # Learning rate for optimizer
GAMMA = 0.99                # Discount factor
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_END = 0.01          # Final exploration rate
EPSILON_DECAY = 0.995       # Epsilon decay rate per episode
BATCH_SIZE = 64             # Mini-batch size for training
BUFFER_CAPACITY = 10000     # Replay buffer capacity
TARGET_UPDATE_FREQ = 10     # Update target network every N episodes

# Network Architecture
HIDDEN_DIMS = [64, 64]      # Hidden layer dimensions

# Evaluation Configuration
EVAL_EPISODES = 100         # Number of episodes for evaluation

# Visualization Configuration
MOVING_AVG_WINDOW = 50      # Window size for moving average in plots

# Advanced Settings
RANDOM_SEED = 42            # Random seed for reproducibility (None for random)
USE_CUDA = True             # Set to True to use GPU (CUDA) for training

# Experiment Configurations
# Uncomment to try different settings:

# Configuration 1: More Fairness-Focused
# LAMBDA_FAIRNESS = 0.8
# NUM_EPISODES = 1500

# Configuration 2: More Accuracy-Focused
# LAMBDA_FAIRNESS = 0.2
# NUM_EPISODES = 1000

# Configuration 3: Larger Network
# HIDDEN_DIMS = [128, 128]
# LEARNING_RATE = 0.0005

# Configuration 4: Longer Training
# NUM_EPISODES = 2000
# MAX_STEPS = 50000
# EPSILON_DECAY = 0.997
