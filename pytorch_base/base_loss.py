from typing import List, Dict, Any, Tuple
import torch

class BaseLoss:
    """
    Base class for defining loss functions in a PyTorch project.

    This class serves as a template for computing losses and other statistics
    that need to be logged during training and evaluation. Subclasses should
    implement the `compute_loss` method and optionally `log_epoch_summary`.
    """
    def __init__(self, stats_names: List[str]):
        """
        Initializes the BaseLoss class.

        :param stats_names: A list of strings representing the names of the statistics
                            to be computed and logged during each call to `compute_loss`.
                            These names will be used as keys in the dictionary returned
                            by `compute_loss`.
        """
        self.stats_names: List[str] = stats_names

    def log_epoch_summary(self, instance: Any, model: torch.nn.Module, epoch: int) -> None:
        """
        Optional method to log a summary at the end of an epoch.

        This method can be implemented by subclasses to perform additional logging
        or computations that are specific to an epoch, such as generating visualizations
        or calculating epoch-level metrics.

        :param instance: A batched instance from the dataloader. This could be a single
                         batch or a specific subset of data used for summary generation.
        :param model: The PyTorch model being trained or evaluated.
        :param epoch: The current epoch number.
        :return: None
        """
        # Default implementation does nothing. Subclasses can override this.
        pass

    def compute_loss(self, instance: Any, model: torch.nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the loss and other specified statistics for a given batch of data.

        This is the primary method that subclasses must implement. It takes a batch
        of data and the model, performs a forward pass, calculates the loss,
        and computes any other statistics defined in `self.stats_names`.

        :param instance: The batched instance from the dataloader. This typically
                         contains input features and target labels.
        :param model: The PyTorch model to be used for the forward pass.
        :return: A tuple containing:
                 - The aggregated loss tensor for backpropagation. This should be a
                   scalar tensor.
                 - A dictionary where keys are the stat names (from `self.stats_names`)
                   and values are their corresponding computed values (as floats).
                   Example: {"loss": 1.5, "accuracy": 0.98}
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        # Example implementation (subclasses should override this):
        # output = model(instance['data'])
        # loss = some_criterion(output, instance['labels'])
        # stats = {"loss": loss.item(), "accuracy": (torch.argmax(output, dim=1) == instance['labels']).float().mean().item()}
        # return loss, stats
        raise NotImplementedError("Subclasses must implement the compute_loss method.")
        # return loss, {"loss1":1.5, "loss2": 0.4, "accuracy":98.5} # Placeholder