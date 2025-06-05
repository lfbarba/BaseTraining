# pytorch-base

[![PyPI version](https://badge.fury.io/py/pytorch-base.svg)](https://badge.fury.io/py/pytorch-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

`pytorch-base` is a lightweight, extensible base framework designed to streamline PyTorch training and experimentation workflows. It provides a structured approach to organizing training code, managing experiments, tracking statistics, and handling common boilerplate tasks associated with PyTorch projects.

The main purpose is to accelerate research and development by allowing users to focus more on model architecture and custom logic, rather than repetitive setup and training loop management.

## Key Features

*   **Simplified Training and Evaluation Loops:** Abstracted training and testing procedures within `PyTorchExperiment`.
*   **Easy Customization of Loss Functions:** Define custom loss computations and metrics by inheriting from `BaseLoss`.
*   **Flexible Statistics Tracking:** Utilize `StatsTracker` for detailed metric tracking during training and evaluation.
*   **Weights & Biases Integration:** Optional, seamless integration with `wandb` for experiment logging and visualization.
*   **Model Checkpointing:** Automatic saving of the best performing models based on a specified metric.
*   **Clear Project Structure:** Encourages a clean and organized way to set up PyTorch projects.
*   **Reproducibility:** Facilities for setting random seeds.

## Installation

### Using pip

Once published to PyPI, you can install `pytorch-base` using pip:
```bash
pip install pytorch-base
```
For local development or to install from a cloned repository:
```bash
pip install .
```

### Using Conda

You can create a Conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate pytorch-base-env
```
This will install all necessary dependencies, including PyTorch.

## Getting Started / Usage Example

Here's a quick example to demonstrate how to use `pytorch-base`:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss
from typing import Tuple, Dict, Any

# 1. Define a simple PyTorch Dataset (or use your own)
class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000, input_features=64, output_features=32):
        self.input_data = torch.randn(num_samples, input_features)
        self.target_data = torch.randn(num_samples, output_features)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.target_data[idx]

# 2. Create a custom Loss class by inheriting from BaseLoss
class MyCustomLoss(BaseLoss):
    def __init__(self):
        # Define the names of the statistics you want to track
        super().__init__(stats_names=["loss", "mse_custom"])
        self.mse_loss = nn.MSELoss()

    def compute_loss(self, instance: Tuple[torch.Tensor, torch.Tensor], model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        data, target = instance  # Unpack the instance from DataLoader

        # Forward pass
        output = model(data)

        # Compute aggregated loss for backpropagation
        loss = self.mse_loss(output, target)

        # Compute other statistics
        stats = {
            "loss": loss.item(), # .item() is important for logging scalar values
            "mse_custom": loss.item() # Example of another stat
        }
        return loss, stats

# 3. Define your PyTorch model
model = nn.Linear(in_features=64, out_features=32)

# 4. Prepare datasets
train_dataset = SimpleDataset(num_samples=1000)
test_dataset = SimpleDataset(num_samples=200)

# 5. Instantiate PyTorchExperiment
# 'args' can be an argparse Namespace or any object with experiment configurations
class DummyArgs:
    def __init__(self):
        self.some_hyperparam = 10

args = DummyArgs()

experiment = PyTorchExperiment(
    args=args, # For wandb logging
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    model=model,
    loss_fn=MyCustomLoss(),
    checkpoint_path="best_model.pt",
    experiment_name="my_pytorch_experiment", # Used for wandb project name
    with_wandb=False, # Set to True to enable wandb
    seed=42,
    loss_to_track="loss", # Metric to monitor for best model checkpointing
    verbose=True
)

# 6. Define an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

# 7. Run the training
# Milestones are epochs for LR decay, gamma is the decay factor.
# The default scheduler steps per batch.
experiment.train(epochs=10, optimizer=optimizer, milestones=[5, 8], gamma=0.1)

print("Training finished. Checkpoint saved to best_model.pt if validation loss improved.")
```

## Core Components

*   **`PyTorchExperiment`**: This is the main class that orchestrates the training and evaluation lifecycle. It handles dataloading, epoch iteration, calls to the loss function, optimizer steps, learning rate scheduling, checkpointing, and `wandb` logging.
*   **`BaseLoss`**: An abstract base class that users must inherit from to define their specific loss calculations and any other metrics they wish to track. The `compute_loss` method is where the core logic resides.
    ```python
    class YourLoss(BaseLoss):
        def __init__(self):
            super().__init__(stats_names=["loss", "accuracy", "custom_metric"])
            # ... initialize your criteria ...

        def compute_loss(self, instance, model):
            # ... your forward pass, loss calculation, and stats dictionary ...
            # return aggregated_loss_tensor, stats_dict
            pass
    ```
*   **`StatsTracker`**: A utility class used internally by `PyTorchExperiment` (one for training, one for testing) to accumulate and average statistics over batches within an epoch. It handles console logging and `wandb` logging of these statistics.

## Running the Example

1.  Ensure you have `pytorch-base` and its dependencies installed (see [Installation](#installation)).
2.  Save the code from the [Getting Started / Usage Example](#getting-started--usage-example) section into a Python file (e.g., `run_example.py`).
3.  Execute the script from your terminal:
    ```bash
    python run_example.py
    ```

## Extending `BaseLoss`

To create your custom loss function and metric tracking:

1.  Create a new class that inherits from `pytorch_base.BaseLoss`.
2.  In the `__init__` method of your custom loss class:
    *   Call `super().__init__(stats_names=[...])`, providing a list of strings that are the names of all statistics you intend to calculate and log (e.g., `"loss"`, `"accuracy"`, `"mae"`).
    *   Initialize any loss criteria (e.g., `nn.CrossEntropyLoss()`, `nn.MSELoss()`) you'll need.
3.  Implement the `compute_loss(self, instance, model)` method:
    *   `instance`: This is the batched output from your DataLoader. Unpack it as needed (e.g., `data, targets = instance`).
    *   `model`: This is your PyTorch model.
    *   Perform the forward pass: `predictions = model(data)`.
    *   Calculate the primary loss value that will be used for backpropagation (e.g., `aggregated_loss = criterion(predictions, targets)`).
    *   Create a dictionary (e.g., `stats_dict`) where keys are the strings you defined in `stats_names` and values are the corresponding *scalar* Python numbers for those statistics (e.g., `loss.item()`, `accuracy_metric.item()`).
    *   Return the `aggregated_loss` (as a PyTorch tensor) and the `stats_dict`.

The `PyTorchExperiment` class will automatically use your custom loss class to compute losses and `StatsTracker` will log the metrics you define.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (once created).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or find bugs.
