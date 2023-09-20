import torch
from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

if __name__ == '__main__':
    trainSet = [(x, y) for x, y in zip(torch.randn(1000, 64), torch.randn(1000, 32))]
    testSet = [(x, y) for x, y in zip(torch.randn(1000, 64), torch.randn(100, 32))]


    class MyLoss(BaseLoss):
        def __init__(self):
            stats_names = ["loss"]
            super(MyLoss, self).__init__(stats_names)

        def compute_loss(self, instance, model):
            mse = torch.nn.MSELoss()
            data, target = instance
            output = model(data)
            loss = mse(output, target)
            return loss, {"loss": loss}


    model = torch.nn.Linear(64, 32)

    exp = PyTorchExperiment(
        train_dataset=trainSet,
        test_dataset=testSet,
        batch_size=20,
        model=model,
        loss_fn=MyLoss(),
        checkpoint_path="checkpoint.pt",
        experiment_name="test_experiment",
        num_workers=0,
        with_wandb = False,
        seed=0,
        loss_to_track = "loss"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001)

    exp.train(10, optimizer, milestones=[5, 2], gamma=0.1)