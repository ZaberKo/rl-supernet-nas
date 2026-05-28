# %%
import json
import matplotlib.pyplot as plt
import numpy as np

file_path = "runs/atari_space_invaders/stage1_eval_archs/eval_records.jsonl"

params = []
returns = []
returns_std = []

with open(file_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        params.append(data["policy_params"])
        returns.append(data["return"])
        returns_std.append(data["return_std"])

params = np.array(params)
returns = np.array(returns)
returns_std = np.array(returns_std)

# Sort by parameters for better visualization
sorted_indices = np.argsort(params)
params = params[sorted_indices]
returns = returns[sorted_indices]
returns_std = returns_std[sorted_indices]

# %%
plt.figure(figsize=(10, 6))
plt.plot(params, returns, 'o-', markersize=4, label='Return')
plt.fill_between(params, returns - returns_std, returns + returns_std, alpha=0.2, label='± 1 Std Dev')

plt.xlabel("Policy Parameters")
plt.ylabel("Return")
plt.title("Policy Parameters vs Return")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
