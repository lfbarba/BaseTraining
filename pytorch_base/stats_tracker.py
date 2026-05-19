import torch
import wandb
from typing import List, Dict, Union, Optional

class StatsTracker:
    """
    A utility class for tracking and logging various statistics (e.g., loss, accuracy)
    over epochs during model training and evaluation.

    It accumulates values for specified statistics on a per-batch basis and can
    compute means. It also supports logging to the console and Weights & Biases (wandb).
    """
    def __init__(self, log_prefix: str, stat_names: List[str], with_wandb: bool = False):
        """
        Initializes the StatsTracker.

        :param log_prefix: A string prefix to be used for console logging (e.g., "Train", "Test").
                           This helps differentiate logs when using multiple trackers.
        :param stat_names: A list of strings representing the names of the statistics
                           to be tracked (e.g., ["loss", "accuracy"]).
        :param with_wandb: Boolean flag to enable/disable Weights & Biases logging for this tracker.
        """
        self.log_prefix: str = log_prefix
        # Stores stats like: {'loss': {'total': tensor(0.0), 'count': 0}, 'accuracy': {'total': tensor(0.0), 'count': 0}}
        self.stats: Dict[str, Dict[str, Union[torch.Tensor, int]]] = {
            name: {'total': torch.tensor(0.0, dtype=torch.float32), 'count': 0}
            for name in stat_names
        }
        self.with_wandb: bool = with_wandb

    def add(self, stat_value_dict: Dict[str, Union[float, torch.Tensor]], batch_size: int) -> None:
        """
        Adds new values for the tracked statistics for a single batch.

        The `stat_value_dict` should contain values for the statistics that were
        defined during initialization.

        :param stat_value_dict: A dictionary where keys are stat names and values are
                                the corresponding stat values for the current batch.
                                Values can be Python floats or single-element PyTorch tensors.
        :param batch_size: The number of samples in the current batch. This is used
                           to correctly weight the statistics if batches have varying sizes.
        :raises ValueError: If a stat_name in `stat_value_dict` was not defined during initialization.
        """
        for stat_name, value in stat_value_dict.items():
            if stat_name not in self.stats:
                raise ValueError(f"Stat name '{stat_name}' not defined for this tracker. Defined stats are: {list(self.stats.keys())}")

            # Ensure value is a float or a scalar tensor
            current_value: float
            if isinstance(value, torch.Tensor):
                current_value = value.item() # Extract float from tensor
            elif isinstance(value, (float, int)):
                current_value = float(value)
            else:
                raise TypeError(f"Stat value for '{stat_name}' must be a float or a tensor, got {type(value)}")

            self.stats[stat_name]['total'] += current_value * batch_size # type: ignore
            self.stats[stat_name]['count'] += batch_size # type: ignore

    def get_mean(self, stat_name: str) -> float:
        """
        Calculates the mean value for a given statistic accumulated so far.

        :param stat_name: The name of the statistic for which to compute the mean.
        :return: The mean value of the statistic. Returns 0.0 if no data has been added
                 for this statistic (to avoid division by zero).
        :raises ValueError: If `stat_name` was not defined during initialization.
        """
        if stat_name not in self.stats:
            raise ValueError(f"Stat name '{stat_name}' not defined for this tracker. Defined stats are: {list(self.stats.keys())}")

        total = self.stats[stat_name]['total']
        count = self.stats[stat_name]['count']

        if count == 0:
            return 0.0  # Avoid division by zero if no stats were added
        return (total / count).item() if isinstance(total, torch.Tensor) else float(total / count)

    def log_stats_and_reset(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Logs the mean values of all tracked statistics to the console and (if enabled)
        to Weights & Biases. After logging, it resets the internal state of the tracker
        (totals and counts) to prepare for the next logging period (e.g., the next epoch).

        :param epoch: Optional. The current epoch number. If provided, it's included in console logs
                      and used as the step for wandb logging.
        :return: A dictionary containing the mean values of all tracked stats, with stat names as keys.
        """
        logged_means: Dict[str, float] = {}
        log_entry_parts: List[str] = []

        if self.log_prefix:
            log_entry_parts.append(f"[{self.log_prefix}]")
        if epoch is not None:
            log_entry_parts.append(f"Epoch {epoch}")

        wandb_log_dict: Dict[str, float] = {}

        for stat_name in self.stats.keys():
            mean_stat = self.get_mean(stat_name)
            logged_means[stat_name] = mean_stat
            # Add to console log parts
            log_entry_parts.append(f"({stat_name}) Mean: {mean_stat:.4f}")

            # Prepare log entry for wandb
            if self.with_wandb and wandb.run:
                # Construct a unique name for wandb, e.g., "Train_loss_mean" or "Test_accuracy_mean"
                wandb_key = f"{self.log_prefix}/{stat_name}_mean" if self.log_prefix else f"{stat_name}_mean"
                wandb_log_dict[wandb_key] = mean_stat

        # Print consolidated log entry to console
        if log_entry_parts: # Ensure there's something to print
            print(" | ".join(log_entry_parts))


        # Log to wandb if enabled and dictionary is not empty
        if self.with_wandb and wandb.run and wandb_log_dict:
            if epoch is not None:
                wandb.log(wandb_log_dict, step=epoch)
            else:
                wandb.log(wandb_log_dict)


        # Reset counters for the next period
        for key in self.stats:
            self.stats[key]['total'] = torch.tensor(0.0, dtype=torch.float32)
            self.stats[key]['count'] = 0

        return logged_means

# Example Usage (can be uncommented for testing):
# if __name__ == "__main__":
#     # Initialize tracker
#     tracker = StatsTracker(log_prefix="Train", stat_names=['loss', 'accuracy'], with_wandb=False) # Set with_wandb=True if testing wandb
#
#     # Simulate adding stats for a few batches in epoch 0
#     tracker.add({'loss': 1.2, 'accuracy': 0.6}, batch_size=32)
#     tracker.add({'loss': torch.tensor(1.1), 'accuracy': torch.tensor(0.65)}, batch_size=32)
#     tracker.add({'loss': 1.0, 'accuracy': 0.7}, batch_size=16) # Different batch size
#
#     # Log stats at the end of epoch 0
#     epoch0_means = tracker.log_stats_and_reset(epoch=0)
#     print(f"Returned means for epoch 0: {epoch0_means}")
#
#     # Simulate adding stats for a few batches in epoch 1
#     tracker.add({'loss': 0.8, 'accuracy': 0.75}, batch_size=32)
#     tracker.add({'loss': 0.7, 'accuracy': 0.8}, batch_size=32)
#
#     # Log stats at the end of epoch 1
#     epoch1_means = tracker.log_stats_and_reset(epoch=1)
#     print(f"Returned means for epoch 1: {epoch1_means}")
#
#     # Test get_mean directly (after reset, it should be 0 or based on new data)
#     print(f"Mean loss after reset (before new data for epoch 2): {tracker.get_mean('loss'):.4f}") # Should be 0.0
#
#     # Test adding a stat not in stat_names (should raise ValueError)
#     try:
#         tracker.add({'new_metric': 0.5}, batch_size=32)
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
#
#     # Test get_mean for a stat not in stat_names (should raise ValueError)
#     try:
#         tracker.get_mean('unknown_metric')
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
