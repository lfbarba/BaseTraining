from contextlib import nullcontext
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
                 args,
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
                 loss_to_track: str = "loss",
                 save_always:bool = False,
                 verbose: bool = True,
                 is_logging_epoch = lambda x:True,
                 mixed_precision=False,
                 amp_dtype: str = "float16",
                 enable_tf32: bool = False,
                 cudnn_benchmark: bool = False,
                 compile_model: bool = False,
                 compile_kwargs=None,
                 pin_memory: bool = False,
                 persistent_workers: bool = False,
                 prefetch_factor=None,
                 ):
        self.train_loader = self._build_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        self.test_loader = self._build_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        self._checkpoint_model = model
        self.model = self._compile_model_if_requested(model, compile_model, compile_kwargs)
        self.amp_device_type = self._infer_model_device(self._checkpoint_model).type
        self.use_mixed_precision, self.amp_dtype = self._resolve_amp_config(mixed_precision, amp_dtype)
        self.grad_scaler = self._build_grad_scaler()
        self._configure_torch_backends(enable_tf32, cudnn_benchmark)
        self.seed = seed
        self.loss_to_track = loss_to_track
        self.save_always = save_always
        self.verbose = verbose
        torch.manual_seed(seed)
        random.seed(seed)
        self.loss_fn = loss_fn
        self.checkpoint_path = checkpoint_path
        self.with_wandb = with_wandb
        self.best_val_loss = float('inf')
        self.is_logging_epoch = is_logging_epoch
        if with_wandb and experiment_name != "":
            wandb.init(project=experiment_name, name=experiment_name + str(seed), config=args)
            wandb.watch(model)
        elif experiment_name == "":
            experiment_name = f"exp_{random.randint(0, 100000)}"
        self.experiment_name = experiment_name

    @staticmethod
    def _build_loader(
            dataset,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor):
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            # Pinned host memory lets CUDA copies run asynchronously when the
            # loss function uses non_blocking=True on tensor.to(...).
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    @staticmethod
    def _infer_model_device(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_amp_config(self, mixed_precision, amp_dtype):
        if isinstance(mixed_precision, str):
            mode = mixed_precision.lower()
        elif mixed_precision:
            if amp_dtype == torch.float16:
                mode = "fp16"
            elif amp_dtype == torch.bfloat16:
                mode = "bf16"
            else:
                mode = str(amp_dtype).lower()
        else:
            mode = "no"

        if mode in {"no", "none", "false", "off", "0"}:
            return False, None

        if mode == "auto":
            # bf16 is more numerically forgiving when the GPU supports it;
            # otherwise fp16 plus GradScaler is the usual CUDA fast path.
            if (
                    self.amp_device_type == "cuda"
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()):
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif mode in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif mode in {"fp16", "float16", "half", "16", "true"}:
            dtype = torch.float16
        else:
            raise ValueError(
                "mixed_precision must be one of: no, fp16, bf16, auto, True, or False"
            )

        if self.amp_device_type == "cpu" and dtype == torch.float16:
            print("mixed_precision='fp16' requested on CPU; using full precision instead.")
            return False, None
        if (
                self.amp_device_type == "cuda"
                and dtype == torch.bfloat16
                and (
                    not hasattr(torch.cuda, "is_bf16_supported")
                    or not torch.cuda.is_bf16_supported())):
            print("mixed_precision='bf16' requested, but this CUDA device does not support bf16; using full precision instead.")
            return False, None
        return True, dtype

    def _build_grad_scaler(self):
        # Grad scaling is only needed for fp16; bf16 has enough exponent range
        # that scaling usually adds overhead without improving stability.
        enabled = (
            self.use_mixed_precision
            and self.amp_dtype == torch.float16
            and self.amp_device_type == "cuda"
        )
        try:
            return torch.amp.GradScaler(self.amp_device_type, enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    @staticmethod
    def _configure_torch_backends(enable_tf32, cudnn_benchmark):
        if torch.cuda.is_available():
            # TF32 keeps the public dtype as float32 while allowing Tensor Cores
            # to accelerate matmul/convolution-heavy models on Ampere+ GPUs.
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            torch.backends.cudnn.allow_tf32 = enable_tf32
        # Benchmarking lets cuDNN choose faster kernels for fixed input shapes.
        torch.backends.cudnn.benchmark = cudnn_benchmark

    @staticmethod
    def _compile_model_if_requested(model, compile_model, compile_kwargs):
        if not compile_model:
            return model
        if not hasattr(torch, "compile"):
            print("torch.compile requested, but this PyTorch build does not provide it.")
            return model
        try:
            print("Compiling model with torch.compile; the first iteration may be slower.")
            return torch.compile(model, **(compile_kwargs or {}))
        except Exception as e:
            print("torch.compile could not wrap the model, using eager mode instead:", e)
            return model

    def _autocast_context(self):
        if not self.use_mixed_precision:
            return nullcontext()
        try:
            return torch.amp.autocast(
                device_type=self.amp_device_type,
                dtype=self.amp_dtype,
                enabled=True,
            )
        except AttributeError:
            if self.amp_device_type == "cuda":
                return torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True)
            return nullcontext()

    @staticmethod
    def _batch_size(instance):
        if isinstance(instance, (tuple, list)):
            return len(instance[0])
        try:
            return len(instance)
        except TypeError:
            return 1

    @staticmethod
    def _zero_grad(optimizer):
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

    def _backward_step(self, loss, optimizer, scheduler):
        if self.grad_scaler.is_enabled():
            scale_before = self.grad_scaler.get_scale()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            # When fp16 gradients overflow, GradScaler skips optimizer.step().
            # Keep the LR schedule aligned with real optimizer updates.
            if self.grad_scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            scheduler.step()

    def train(self, epochs, optimizer, milestones, gamma, scheduler=None):
        train_tracker = StatsTracker("Train", self.loss_fn.stats_names)
        test_tracker = StatsTracker("Test", self.loss_fn.stats_names)

        if scheduler is None:
            scheduler = MultiStepLR(optimizer, milestones=[x * len(self.train_loader.dataset) for x in milestones], gamma=gamma)

        if self.use_mixed_precision:
            print(f"Mixed precision enabled on {self.amp_device_type} with dtype={self.amp_dtype}.")

        for epoch in range(epochs):
            self.model.train()
            iterator = tqdm(self.train_loader, desc="Training Loop", leave=False, disable=not self.verbose)
            for instance in iterator:
                self._zero_grad(optimizer)
                # Autocast keeps numerically sensitive ops in fp32 while using
                # lower precision kernels for supported model operations.
                with self._autocast_context():
                    loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                self._backward_step(loss, optimizer, scheduler)

                bs_instance = self._batch_size(instance)
                train_tracker.add(loss_dict, bs_instance)
                display_loss = loss_dict.get(self.loss_to_track, loss)
                try:
                    display_loss = display_loss.detach().float().item()
                except AttributeError:
                    display_loss = float(display_loss)
                iterator.set_postfix({"loss": f"{display_loss:.6f}"})

            train_tracker.log_stats_and_reset()

            if self.is_logging_epoch(epoch):
                self.model.eval()
                with torch.no_grad():
                    for instance in tqdm(self.test_loader, desc="Training Loop", leave=False, disable=not self.verbose):
                        with self._autocast_context():
                            loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                        bs_instance = self._batch_size(instance)
                        test_tracker.add(loss_dict, bs_instance)

                    self.loss_fn.log_epoch_summary(instance, self.model, epoch)

                    if self.save_always or test_tracker.get_mean(self.loss_to_track) < self.best_val_loss:
                        self.best_val_loss = test_tracker.get_mean(self.loss_to_track)
                        print("saving models at ", self.checkpoint_path)
                        try:
                            checkpoint = {
                                'model_state_dict': self._checkpoint_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            if self.grad_scaler.is_enabled():
                                checkpoint['amp_scaler_state_dict'] = self.grad_scaler.state_dict()
                            torch.save(checkpoint, self.checkpoint_path)
                        except Exception as e:
                            print("model could not be saved, error:", e)

                        # if wandb.run:
                        #     wandb.save(self.checkpoint_path)

                    test_tracker.log_stats_and_reset()

        if self.with_wandb:
            wandb.finish()
