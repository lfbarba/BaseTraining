class BaseLoss:
    def __init__(self, stats_names:list):
        """
        A tamplate for a loss function computation. It should process a batched instance from the dataloader, pass it through the model,
        process it and compute the losses and other stats to be logged
        :param stats_names: The names of the stats to be logged and computed in each call to compute_loss
        """
        self.stats_names = stats_names

    def log_epoch_summary(self, instance, model, epoch):
        """
        Template for the loss computation
        :param instance: The batched instance from the dataloader
        :param model: the model to the forward pass
        :param epoch: The current epoch of the model
        :return: the aggregated loss for back propagation, a dictionary of labeled losses for logging, same names as in self.stats_names
        """
        pass

    def compute_loss(self, instance, model):
        """
        Template for the loss computation
        :param instance: The batched instance from the dataloader
        :param model: the model to the forward pass
        :return: the aggregated loss for back propagation, a dictionary of labeled losses for logging, same names as in self.stats_names
        """
        pass
        # return loss, {"loss1":1.5, "loss2": 0.4, "accuracy":98.5}