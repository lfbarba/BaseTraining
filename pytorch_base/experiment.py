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
                 train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 model: nn.Module,
                 loss_fn: BaseLoss,
                 checkpoint_path: str,
                 experiment_name: str = "",
                 num_workers: int = 0,
                 with_wandb: bool = False,
                 seed=0,
                 loss_to_track: str = "loss"
                 ):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model = model
        self.seed = seed
        self.loss_to_track = loss_to_track
        torch.manual_seed(seed)
        random.seed(seed)
        self.loss_fn = loss_fn
        self.checkpoint_path = checkpoint_path
        self.best_val_loss = float('inf')
        if with_wandb and experiment_name != "":
            wandb.init(project=experiment_name, name=experiment_name + str(seed))
            wandb.watch(model)
        elif experiment_name == "":
            experiment_name = f"exp_{random.randint(0, 100000)}"
        self.experiment_name = experiment_name

    def train(self, epochs, optimizer, milestones, gamma):
        train_tracker = StatsTracker("Train", self.loss_fn.stats_names)
        test_tracker = StatsTracker("Test", self.loss_fn.stats_names)

        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        for epoch in range(epochs):
            self.model.train()
            iterator = tqdm(self.train_loader)
            for instance in iterator:
                optimizer.zero_grad()
                loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                loss.backward()
                optimizer.step()
                bs_instance = len(instance[0]) if type(instance) == tuple else len(instance)
                train_tracker.add(loss_dict, bs_instance)
                iterator.set_postfix({"loss": f"{loss.item():.2f}"})

            train_tracker.log_stats_and_reset()
            scheduler.step()

            self.model.eval()

            with torch.no_grad():
                for instance in tqdm(self.test_loader):
                    loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                    bs_instance = len(instance[0]) if type(instance) == tuple else len(instance)
                    test_tracker.add(loss_dict, bs_instance)

                if test_tracker.get_mean(self.loss_to_track) < self.best_val_loss:
                    self.best_val_loss = test_tracker.get_mean(self.loss_to_track)
                    print("saving model at ", self.checkpoint_path)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, self.checkpoint_path)

                    if wandb.run:
                        wandb.save(self.checkpoint_path)

                test_tracker.log_stats_and_reset()
