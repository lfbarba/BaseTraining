import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.optim import Optimizer
from pytorch_base.stats_tracker import StatsTracker
from pytorch_base.base_loss import BaseLoss
from tqdm import tqdm
import random
from typing import Any, Callable, List, Optional, Dict

class PyTorchExperiment:
    """
    Manages the training and evaluation process for a PyTorch model.

    This class encapsulates the boilerplate code for training loops,
    evaluation, checkpointing, and integration with Weights & Biases (wandb).
    """
    def __init__(self,
                 args: Any,  # Typically a Namespace object from argparse or a custom config object
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 batch_size: int,
                 model: nn.Module,
                 loss_fn: BaseLoss,
                 checkpoint_path: str,
                 experiment_name: str = "",
                 num_workers: int = 0,
                 with_wandb: bool = False,
                 seed: int = 0,
                 loss_to_track: str = "loss", # The key in the loss_dict used for checkpointing decisions
                 save_always: bool = False, # If True, saves checkpoint at every logging epoch, not just on improvement
                 verbose: bool = True, # If True, shows tqdm progress bars
                 is_logging_epoch: Callable[[int], bool] = lambda epoch: True # Function to determine if current epoch should be logged/tested
                 ):
        """
        Initializes the PyTorchExperiment.

        :param args: Arguments or configuration object for the experiment (e.g., from argparse).
                     Used for logging with wandb if `with_wandb` is True.
        :param train_dataset: The dataset for training.
        :param test_dataset: The dataset for testing/validation.
        :param batch_size: Batch size for both training and testing DataLoaders.
        :param model: The PyTorch nn.Module to be trained.
        :param loss_fn: An instance of BaseLoss (or its subclass) to compute loss and stats.
        :param checkpoint_path: Path where the best model checkpoint will be saved.
        :param experiment_name: Name of the experiment. Used for wandb project and run naming.
                                If empty, a random name is generated.
        :param num_workers: Number of worker processes for DataLoaders.
        :param with_wandb: Boolean flag to enable/disable Weights & Biases logging.
        :param seed: Random seed for reproducibility.
        :param loss_to_track: The name of the loss/metric in the dictionary returned by
                              `loss_fn.compute_loss` that should be monitored to determine
                              the "best" model for checkpointing.
        :param save_always: If True, a checkpoint is saved at every epoch where logging occurs.
                            If False (default), a checkpoint is saved only if the `loss_to_track`
                            improves on the test set.
        :param verbose: If True, progress bars (tqdm) will be displayed during training and testing.
        :param is_logging_epoch: A function that takes the current epoch number as input and
                                 returns True if logging and evaluation should be performed
                                 for that epoch, False otherwise. Defaults to logging every epoch.
        """
        self.train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model: nn.Module = model
        self.seed: int = seed
        self.loss_to_track: str = loss_to_track
        self.save_always: bool = save_always
        self.verbose: bool = verbose
        self.is_logging_epoch: Callable[[int], bool] = is_logging_epoch

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)

        self.loss_fn: BaseLoss = loss_fn
        self.checkpoint_path: str = checkpoint_path
        self.with_wandb: bool = with_wandb
        self.best_val_loss: float = float('inf') # Initialize with a very high value

        # Initialize wandb if enabled
        if with_wandb:
            if not experiment_name: # Generate a random experiment name if not provided
                experiment_name = f"exp_{random.randint(0, 100000)}"
            wandb.init(project=experiment_name, name=f"{experiment_name}_seed{seed}", config=args)
            wandb.watch(model) # Log gradients and model topology
        self.experiment_name: str = experiment_name


    def train(self,
              epochs: int,
              optimizer: Optimizer,
              milestones: List[int], # Epoch numbers for LR decay
              gamma: float, # LR decay factor
              scheduler: Optional[_LRScheduler] = None # Optional pre-configured scheduler
              ) -> None:
        """
        Runs the training and evaluation loop.

        :param epochs: Total number of epochs to train.
        :param optimizer: The PyTorch optimizer (e.g., Adam, SGD).
        :param milestones: A list of epoch indices at which the learning rate should be
                           multiplied by `gamma`. Relevant if the default MultiStepLR is used.
        :param gamma: The learning rate decay factor for MultiStepLR.
        :param scheduler: An optional learning rate scheduler. If None, a MultiStepLR
                          scheduler is created using `milestones` and `gamma`.
                          Note: The default scheduler steps per *batch*, not per epoch.
        """
        train_tracker = StatsTracker(log_prefix="Train", stat_names=self.loss_fn.stats_names, with_wandb=self.with_wandb)
        test_tracker = StatsTracker(log_prefix="Test", stat_names=self.loss_fn.stats_names, with_wandb=self.with_wandb)

        # Initialize learning rate scheduler if not provided
        if scheduler is None:
            # Note: This scheduler steps per batch.
            # Convert epoch milestones to iteration milestones
            iteration_milestones = [m * len(self.train_loader) for m in milestones]
            scheduler = MultiStepLR(optimizer, milestones=iteration_milestones, gamma=gamma)

        for epoch in range(epochs):
            self.model.train() # Set model to training mode
            # Setup progress bar for the training loop
            iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False, disable=not self.verbose)
            for instance in iterator:
                optimizer.zero_grad() # Clear previous gradients

                # Forward pass and loss computation
                loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)

                loss.backward() # Backpropagate the loss
                optimizer.step() # Update model parameters
                scheduler.step() # Update learning rate

                # Determine batch size for accurate statistics (handles tuples or single tensors)
                current_batch_size = len(instance[0]) if isinstance(instance, (list, tuple)) and len(instance) > 0 else instance.size(0) if torch.is_tensor(instance) else 1
                train_tracker.add(loss_dict, current_batch_size) # Add batch stats to tracker
                iterator.set_postfix({"loss": f"{loss.item():.6f}"}) # Update progress bar

            # Log training stats for the epoch and reset tracker
            train_tracker.log_stats_and_reset(epoch=epoch)

            # Evaluation phase (if current epoch is a logging epoch)
            if self.is_logging_epoch(epoch):
                self.model.eval() # Set model to evaluation mode
                with torch.no_grad(): # Disable gradient calculations for efficiency
                    # Setup progress bar for the testing loop
                    test_iterator = tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{epochs} [Testing]", leave=False, disable=not self.verbose)
                    for instance in test_iterator:
                        # Forward pass and loss computation
                        loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                        current_batch_size = len(instance[0]) if isinstance(instance, (list, tuple)) and len(instance) > 0 else instance.size(0) if torch.is_tensor(instance) else 1
                        test_tracker.add(loss_dict, current_batch_size) # Add batch stats to tracker
                        test_iterator.set_postfix({"loss": f"{loss.item():.6f}"})


                    # Allow the loss function to perform additional epoch-level logging (e.g., visualizations)
                    # Pass the last instance from test_loader for potential use in visualization
                    if len(self.test_loader) > 0:
                        last_test_instance = instance # or next(iter(self.test_loader)) if preferred
                        self.loss_fn.log_epoch_summary(last_test_instance, self.model, epoch)

                    # Checkpoint saving logic
                    current_tracked_loss = test_tracker.get_mean(self.loss_to_track)
                    if self.save_always or current_tracked_loss < self.best_val_loss:
                        self.best_val_loss = current_tracked_loss
                        if self.verbose:
                            print(f"Epoch {epoch+1}: New best validation {self.loss_to_track}: {self.best_val_loss:.6f}. Saving model to {self.checkpoint_path}")
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss': self.best_val_loss,
                                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            }, self.checkpoint_path)
                            # If using wandb, save the checkpoint file to wandb as an artifact
                            # if self.with_wandb and wandb.run:
                            #     wandb.save(self.checkpoint_path) # This saves the file directly
                        except Exception as e:
                            print(f"Model could not be saved due to error: {e}")

                # Log testing stats for the epoch and reset tracker
                test_tracker.log_stats_and_reset(epoch=epoch)

        # Finalize wandb run if it was initialized
        if self.with_wandb and wandb.run is not None:
            wandb.finish()
