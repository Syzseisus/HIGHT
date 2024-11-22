import math
import numpy as np

import torch
from torch import optim
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR, CosineAnnealingLR

from utils import rank_zero_print


class MoleBERTFT_pl(pl.LightningModule):
    def __init__(self, training_args, criterion, gnn, attn_pooling=False):
        super().__init__()
        self.args = training_args
        self.attn_pooling = attn_pooling
        self.model = gnn
        self.criterion = criterion

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

        self.log_kwargs = {
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "sync_dist": True,
        }

        self._bnb_setting()

    def forward(self, batch):
        return self.model(batch)

    def _common_step(self, batch):
        # forward
        pred, _ = self(batch)

        # loss
        y = batch.y.view(pred.shape)
        non_null = y**2 > 0
        loss_raw = self.criterion(pred, (y + 1) * 0.5)
        loss_filt = torch.where(non_null, loss_raw, torch.zeros_like(loss_raw))
        loss = torch.sum(loss_filt) / torch.sum(non_null)

        return loss, pred, y

    def _calculate_epoch_metrics(self, outputs):
        y_true = torch.cat([x["y"] for x in outputs], dim=0).cpu().numpy()
        y_pred = torch.cat([x["pred"] for x in outputs], dim=0).cpu().numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                non_null = y_true[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true[non_null, i] + 1) / 2, y_pred[non_null, i]))

        auc = sum(roc_list) / len(roc_list) if roc_list else 0.0
        return auc

    def on_train_epoch_start(self):
        self.train_step_outputs = []

    def on_val_epoch_start(self):
        self.val_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch)
        self.train_step_outputs.append({"pred": pred.detach().cpu(), "y": y.detach().cpu()})
        self.log("train_loss", loss.item(), batch_size=len(batch), prog_bar=True, **self.log_kwargs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch)
        self.val_step_outputs.append({"pred": pred.detach().cpu(), "y": y.detach().cpu(), "loss": loss.item()})
        return loss

    def test_step(self, batch, batch_idx):
        pred, _ = self(batch)
        y = batch.y.view(pred.shape)
        self.test_step_outputs.append({"pred": pred.detach().cpu(), "y": y.detach().cpu()})

    def on_train_epoch_end(self):
        self.trainer.strategy.barrier()
        auc = self._calculate_epoch_metrics(self.train_step_outputs)
        self.log("train_auc", auc, prog_bar=True, logger=True, sync_dist=True)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.trainer.strategy.barrier()

        loss = sum(x["loss"] for x in self.val_step_outputs) / len(self.val_step_outputs)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)

        auc = self._calculate_epoch_metrics(self.val_step_outputs)
        self.log("val_auc", auc, prog_bar=True, logger=True, sync_dist=True)
        self.val_step_outputs.clear()

    def on_test_epoch_end(self):
        auc = self._calculate_epoch_metrics(self.test_step_outputs)
        self.log("test_auc", auc, prog_bar=True, logger=True, sync_dist=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        lr = self.args.learning_rate
        scaled_lr = lr * self.args.lr_scale
        weight_decay = self.args.weight_decay

        param_group = []
        scaled_params = set()

        # Graph prediction layer
        pred_params = set(self.model.graph_pred_linear.parameters())
        scaled_params.update(pred_params)
        param_group.append({"params": list(pred_params), "lr": scaled_lr})

        # Attention pooling layer (if used)
        if self.attn_pooling:
            pool_params = set(self.model.pool.parameters())
            scaled_params.update(pool_params)
            param_group.append({"params": list(pool_params), "lr": scaled_lr})

        # 나머지 파라미터들은 base learning rate 적용
        base_params = [p for p in self.model.parameters() if p not in scaled_params]
        param_group.append({"params": base_params, "lr": lr})

        optimizer = optim.Adam(param_group, weight_decay=weight_decay)

        lr_sched = self.args.lr_scheduler_type
        if not lr_sched:
            return [optimizer]

        # TODO: lr scheduler configure arg로 만들기 (일단 lambda만 쓰기)
        rank_zero_print(f"Use `{lr_sched}` LR scheduler")
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())

        warmup_steps = self.args.warmup_epochs * steps_per_epoch
        max_steps = self.args.num_train_epochs * steps_per_epoch
        if lr_sched == "lambda":
            lr_lambda = lambda step: self.get_lr_lambda(step, warmup_steps, max_steps)
            scheduler = LambdaLR(optimizer, lr_lambda)

        elif lr_sched == "linear":
            scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

        elif lr_sched == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=lr / 1000)

        elif lr_sched == "warmup_cosine":
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=lr / 1000)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        else:
            raise NotImplementedError(f"{lr_sched} is not implemented yet.")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def get_lr_lambda(self, curr_step, warmup_steps, max_steps):
        if curr_step < warmup_steps:
            return float(curr_step) / float(max(1, warmup_steps))
        else:
            progress = float(curr_step - warmup_steps) / float(max(1, (max_steps - warmup_steps)))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _bnb_setting(self):
        """borrow from https://github.com/haotian-liu/LLaVA"""
        compute_dtype = torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else torch.float32)

        bnb_model_from_pretrained_args = {}
        # load 4 or 8 bit
        if self.args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_int8_training

            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=self.args.bits == 4,
                    load_in_8bit=self.args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=self.args.double_quant,
                    bnb_4bit_quant_type=self.args.quant_type,
                ),
            )
            bnb_model_from_pretrained_args["quantization_config"] = quantization_config
            if "," not in self.args.gpus:
                bnb_model_from_pretrained_args["device_map"] = {"": self.args.gpus}
