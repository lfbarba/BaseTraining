import torch
import wandb


class StatsTracker:
    def __init__(self, name, stat_names):
        """
        Initialize the StatsTracker with a list of stat names.

        Parameters:
        - stat_names: List of names of the stats you want to track.
        """
        self.name = name

        self.stats = {name: {'total': torch.tensor(0.0), 'count': 0}
                      for name in stat_names}
        self.current_epoch = 0

    def add(self, stat_value_dict, batch_size):
        """
        Add a new stat value for a given stat_name.

        Parameters:
        - stat_name: The name of the stat you want to add a value for.
        - value: The stat value.
        - batch_size: The batch size.
        - epoch: Current training epoch.
        """
        for stat_name in stat_value_dict:
            if stat_name not in self.stats:
                raise ValueError(f"Stat name {stat_name} not found!")

            self.stats[stat_name]['total'] += stat_value_dict[stat_name].to('cpu') * batch_size
            self.stats[stat_name]['count'] += batch_size

    def get_mean(self, stat_name):
        """
        Get the mean stat value for a given stat_name for the current epoch.

        Parameters:
        - stat_name: The name of the stat you want to retrieve the mean for.

        Returns:
        - Mean stat value.
        """
        if stat_name not in self.stats:
            raise ValueError(f"Stat name {stat_name} not found!")

        return self.stats[stat_name]['total'] / self.stats[stat_name]['count']

    def log_stats_and_reset(self):
        """
        Log the current mean, total, min, and max values of all the tracked stats for the current epoch.
        Also logs the values to wandb if initialized.
        """
        epoch = self.current_epoch
        for stat_name, data in self.stats.items():
            mean_stat = self.get_mean(stat_name)
            print(
                f"[{self.name} Epoch {epoch}] ({stat_name}) Mean: {mean_stat:.2f}")

            # Log to wandb if initialized
            if wandb.run:
                wandb.log({
                    f"{self.name}_{stat_name}_mean": mean_stat,
                }, step=epoch)

        # reset counters for next epoch
        self.current_epoch += 1
        for key in self.stats:
            self.stats[key]['total'] = torch.tensor(0.0)
            self.stats[key]['count'] = 0

# # Example Usage:
# tracker = StatsTracker(['stat1', 'stat2'])
# tracker.add('stat1', torch.tensor(1.2), 32)
# tracker.add('stat2', torch.tensor(0.8), 32)
# tracker.log_stats()
# tracker.add('stat1', torch.tensor(1.1), 32)
# tracker.add('stat2', torch.tensor(0.5), 32)
# tracker.log_stats()
# tracker.add('stat1', torch.tensor(1.91), 32)
# tracker.add('stat2', torch.tensor(0.2), 32)
# tracker.log_stats()
