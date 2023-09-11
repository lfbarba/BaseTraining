import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.stats_tracker import StatsTracker
from losses.base_loss import BaseLoss


class PyTorchExperiment:
    def __init__(self,
                 train_dataset:torch.utils.data.Dataset,
                 test_dataset:torch.utils.data.Dataset,
                 batch_size:int,
                 model:nn.Module,
                 loss_fn:BaseLoss,
                 experiment_name: str = ""
                 ):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model = model
        self.loss_fn = loss_fn
        self.best_val_loss = float('inf')
        if experiment_name != "":
            wandb.init(project=experiment_name)
            wandb.watch(model)



    def train(self, epochs, optimizer, milestones, gamma):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        tracker = StatsTracker(self.loss_fn.stats_names)

        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        for epoch in range(epochs):
            self.model.train()
            for instance in self.train_loader:
                optimizer.zero_grad()
                loss, loss_dict = self.loss_fn(instance, self.model)
                loss.backward()
                optimizer.step()
                bs_instance = len(instance[0]) if type(instance) == tuple else len(instance)
                tracker.add(loss_dict, bs_instance)

            tracker.log_stats()
            scheduler.step()

    def test(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        tracker = StatsTracker(self.loss_fn.stats_names)

        with torch.no_grad():
            for instance in self.test_loader:
                loss, loss_dict = self.loss_fn(instance, self.model)
                bs_instance = len(instance[0]) if type(instance) == tuple else len(instance)
                tracker.add(loss_dict, bs_instance)

            tracker.log_stats()


            if tracker.get_mean("loss") < self.best_val_loss:
                self.best_val_loss = tracker.get_mean("loss")
                torch.save(self.model.state_dict(), "best_model.pt")
                wandb.save("best_model.pt")

        return tracker
