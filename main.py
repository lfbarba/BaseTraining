import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # DataLoader is not used directly here but good for context
from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss
from typing import Tuple, List, Dict, Any

# 1. Define a simple custom PyTorch Dataset
# This demonstrates how data should be structured for the PyTorchExperiment.
class SimpleRegressionDataset(Dataset):
    """A simple dataset for regression tasks."""
    def __init__(self, num_samples: int, input_features: int, output_features: int):
        """
        Generates random data for the dataset.
        :param num_samples: Number of samples in the dataset.
        :param input_features: Number of features for the input data.
        :param output_features: Number of features for the target data.
        """
        self.input_data = torch.randn(num_samples, input_features)
        self.target_data = torch.randn(num_samples, output_features)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.input_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample (input data and target data) from the dataset at the given index.
        :param idx: Index of the sample to retrieve.
        :return: A tuple containing input data and target data.
        """
        return self.input_data[idx], self.target_data[idx]


# 2. Define a custom Loss class by inheriting from BaseLoss
class MyRegressionLoss(BaseLoss):
    """
    A custom loss class for a simple regression task using Mean Squared Error (MSE).
    It demonstrates how to define statistics to be tracked.
    """
    def __init__(self):
        # Define the names of the statistics to be tracked.
        # These names will be used as keys in the dictionary returned by compute_loss
        # and logged by StatsTracker.
        stats_names: List[str] = ["loss", "mse"]
        super().__init__(stats_names=stats_names)

        # Initialize the loss criterion
        self.mse_criterion = nn.MSELoss()

    def compute_loss(self, instance: Tuple[torch.Tensor, torch.Tensor], model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the loss and other specified statistics for a given batch of data.
        :param instance: A tuple (data, target) from the DataLoader.
        :param model: The PyTorch model to be used for the forward pass.
        :return: A tuple containing:
                 - The aggregated loss tensor for backpropagation.
                 - A dictionary of labeled losses/statistics for logging.
        """
        # Unpack the instance from the DataLoader
        data, target = instance

        # Perform the forward pass
        output = model(data)

        # Calculate the primary loss for backpropagation
        loss = self.mse_criterion(output, target)

        # Prepare a dictionary of statistics to be logged.
        # It's important to use .item() to get scalar Python numbers for logging.
        stats_dict: Dict[str, float] = {
            "loss": loss.item(), # The primary loss value
            "mse": loss.item()   # In this case, 'mse' is the same as 'loss'
        }

        return loss, stats_dict


if __name__ == '__main__':
    # --- Configuration ---
    NUM_SAMPLES_TRAIN: int = 1000
    NUM_SAMPLES_TEST: int = 200
    INPUT_FEATURES: int = 64
    OUTPUT_FEATURES: int = 32
    BATCH_SIZE: int = 32 # Reduced from main.py for potentially faster example execution
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0001
    SEED: int = 42

    # --- Dataset Preparation ---
    # Create instances of the custom dataset
    train_dataset = SimpleRegressionDataset(NUM_SAMPLES_TRAIN, INPUT_FEATURES, OUTPUT_FEATURES)
    test_dataset = SimpleRegressionDataset(NUM_SAMPLES_TEST, INPUT_FEATURES, OUTPUT_FEATURES)

    # --- Model Definition ---
    # Define a simple linear model for the regression task
    model: nn.Module = nn.Linear(INPUT_FEATURES, OUTPUT_FEATURES)

    # --- PyTorchExperiment Setup ---
    # The 'args' parameter can be an argparse.Namespace or any object/dict holding hyperparameters.
    # It's used by PyTorchExperiment if wandb is enabled to log hyperparameters.
    # For this example, we'll use a simple dictionary.
    experiment_args: Dict[str, Any] = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "architecture": model.__class__.__name__
    }

    # Instantiate the PyTorchExperiment class
    exp = PyTorchExperiment(
        args=experiment_args,               # Arguments/hyperparameters for logging (e.g., to wandb)
        train_dataset=train_dataset,        # Training dataset
        test_dataset=test_dataset,          # Testing/validation dataset
        batch_size=BATCH_SIZE,              # Batch size for DataLoaders
        model=model,                        # The PyTorch model
        loss_fn=MyRegressionLoss(),         # Custom loss function instance
        checkpoint_path="best_model_main.pt", # Path to save the best model checkpoint
        experiment_name="pytorch_base_example", # Name for the experiment (used by wandb)
        num_workers=0,                      # Number of workers for DataLoaders (0 for main process)
        with_wandb=False,                   # Set to True to enable Weights & Biases logging
        seed=SEED,                          # Random seed for reproducibility
        loss_to_track="loss",               # Stat name from MyRegressionLoss to monitor for best model
        verbose=True                        # Whether to show progress bars (tqdm)
    )

    # --- Optimizer Definition ---
    # Define the optimizer for training
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --- Training ---
    # Run the training process
    # Milestones are epochs at which to decay the learning rate by the factor 'gamma'.
    # The default MultiStepLR scheduler steps per batch.
    print(f"Starting training for {EPOCHS} epochs...")
    exp.train(
        epochs=EPOCHS,
        optimizer=optimizer,
        milestones=[5, 8],  # Epochs to decay LR (will be converted to iteration milestones)
        gamma=0.1           # LR decay factor
    )

    print("\nTraining finished.")
    print(f"The best model checkpoint might be saved at 'best_model_main.pt' if validation performance improved.")
    print("To enable Weights & Biases, set 'with_wandb=True' in PyTorchExperiment and ensure you are logged in to wandb.")
    print("You can inspect the 'MyRegressionLoss' and 'SimpleRegressionDataset' classes to see how to customize them.")
    print("The 'pytorch_base.experiment.PyTorchExperiment' class handles the training loop, evaluation, and checkpointing.")
    print("The 'pytorch_base.base_loss.BaseLoss' is the parent class for custom loss functions.")
    print("The 'pytorch_base.stats_tracker.StatsTracker' is used internally to track and log metrics.")

# To run this example:
# 1. Ensure pytorch-base is installed (e.g., `pip install .` from the project root).
# 2. Execute this script: `python main.py`
# 3. To use Weights & Biases:
#    - Install wandb: `pip install wandb`
#    - Login: `wandb login`
#    - Set `with_wandb=True` in the `PyTorchExperiment` arguments.
#    - Run the script again. Your experiment will appear on your wandb dashboard.
# 4. The `checkpoint.pt` file (or `best_model_main.pt` as named here) will store the model state_dict
#    of the best performing model on the test set (based on 'loss_to_track').
# 5. The `milestones` for the scheduler are epoch-based but are converted to iteration-based
#    for the default MultiStepLR scheduler which steps per batch.
#    If you want epoch-based stepping for the scheduler, you'd need to provide a custom scheduler instance
#    to `exp.train()` and manage its stepping accordingly (e.g., by not having `scheduler.step()` in the
#    inner loop of `PyTorchExperiment.train` or by providing a scheduler that expects `scheduler.step()`
#    to be called after each epoch). The current setup in `PyTorchExperiment` calls `scheduler.step()` per batch.
#    For simplicity, the default iteration-based stepping is used here.
