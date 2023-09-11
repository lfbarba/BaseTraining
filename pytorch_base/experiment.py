import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_base.stats_tracker import StatsTracker
from pytorch_base.base_loss import BaseLoss
from tqdm import tqdm
import random

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
        else:
            experiment_name = f"exp_{random.randint(0, 100000)}"
        self.experiment_name = experiment_name



    def train(self, epochs, optimizer, milestones, gamma):
        train_tracker = StatsTracker("Train", self.loss_fn.stats_names)
        test_tracker = StatsTracker("Test", self.loss_fn.stats_names)

        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        for epoch in range(epochs):
            self.model.train()
            for instance in tqdm(self.train_loader):
                optimizer.zero_grad()
                loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                loss.backward()
                optimizer.step()
                bs_instance = len(instance[0]) if type(instance) == tuple else len(instance)
                train_tracker.add(loss_dict, bs_instance)

            train_tracker.log_stats_and_reset()
            scheduler.step()

            self.model.eval()

            with torch.no_grad():
                for instance in tqdm(self.test_loader):
                    loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                    bs_instance = len(instance[0]) if type(instance) == tuple else len(instance)
                    test_tracker.add(loss_dict, bs_instance)

                if test_tracker.get_mean("loss") < self.best_val_loss:
                    self.best_val_loss = test_tracker.get_mean("loss")
                    torch.save(self.model.state_dict(), f"{self.experiment_name}.pt")
                    if wandb.run:
                        wandb.save(f"snapshots/{self.experiment_name}.pt")

                test_tracker.log_stats_and_reset()
