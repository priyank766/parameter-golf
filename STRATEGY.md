# Parameter Golf: Hackathon Strategy Document

## Objective
Beat the current 1.1194 baseline on the Parameter Golf leaderboard under the strict constraints of:
- **16MB Zipped File Size** (Code + Weights)
- **<10 minute run-time on 8xH100**

## 1v1 Testing Plan
To mathematically prove our improvements, we will maintain two separate training scripts:
1. `train_leaderboard.py`: Based on the Leaderboard #1 run (Legal Score-First TTT, Parallel Muon). We will stack our mathematical ideas underneath it (like Depth Recurrence or modified activations) to push the score even lower.
2. `train_turboquant.py`: A ground-up implementation specifically targeting the ideas from the Google Research "TurboQuant" blog. We will implement PolarQuant and the QJL (1-bit Johnson-Lindenstrauss) tricks natively in PyTorch for this script.

## The Winning Hacks (Advanced Techniques)

### 1. Depth Recurrence & Parameter Tying (Implemented)
Instead of storing 12 distinct `Attention` and `MLP` layers (which eats up massive MBs), we build ultra-wide blocks and reuse their weights multiple times during the forward pass. 
- *Current Status*: `train_advanced.py` incorporates a `3x MLP` scaling.

### 2. LeakyReLU(0.5)² (Implemented)
A math shortcut replacing the standard activation function, eliminating "dead neurons" and dramatically boosting learning speed.

### 3. Extreme Quantization: PolarQuant + QJL (Planned)
"https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/"
Following Google's recent TurboQuant paper, instead of storing 16-bit float weights, we will compress our model weights into mathematical Polar Coordinates (Angles + Radius).
- We can crush parameter angles down to 3 bits.
- We fix rounding math errors using a 1-bit (-1 / +1) Quantized Johnson-Lindenstrauss trick.
- Explodes our "parameter capacity" from 8 Million to over 35+ Million PyTorch weights strictly capped at 16MB.

### 4. Legal Test-Time Training (TTT) (Planned)
The challenge rules forbid training on validation data *before* evaluation. However, we will build an evaluation loop that evaluates a sequence, logs the score, and *immediately trains on it* before evaluating the next sequence!
