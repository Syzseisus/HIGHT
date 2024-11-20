import math
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
import lightning.pytorch as pl
from utils import rank_zero_print


class GraphVQVAE(nn.Module):
    def __init__(self, criterion, encoder, vqvae_layer, atom_decoder, chiral_decoder, bond_decoder=None):
        super().__init__()
        self.criterion = criterion
        self.encoder = encoder
        self.vqvae_layer = vqvae_layer
        self.atom_decoder = atom_decoder
        self.chiral_decoder = chiral_decoder
        self.bond_decoder = bond_decoder

    def forward(self, batch):
        # forward
        node_rep = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        e, loss_dict = self.vqvae_layer(batch.x, node_rep)
        pred_atom = self.atom_decoder(e, batch.edge_index, batch.edge_attr)
        pred_chiral = self.chiral_decoder(e, batch.edge_index, batch.edge_attr)

        # loss
        atom_loss = self.criterion(pred_atom, batch.x[:, 0])
        chiral_loss = self.criterion(pred_chiral, batch.x[:, 1])
        recon_loss = atom_loss + chiral_loss
        loss_dict["atom_recon_loss"] = atom_loss
        loss_dict["chiral_recon_loss"] = chiral_loss
        loss_dict["total_recon_loss"] = recon_loss

        # forward and loss w.r.t bond
        if self.bond_decoder is not None:
            edge_rep = e[batch.edge_index[0]] + e[batch.edge_index[1]]
            pred_edge = self.bond_decoder(edge_rep, batch.edge_index, batch.edge_attr)
            bond_loss = self.criterion(pred_edge, batch.edge_attr[:, 0])
            recon_loss += bond_loss
            loss_dict["bond_recon_loss"] = bond_loss
            loss_dict["total_recon_loss"] = recon_loss

        # total loss
        loss_dict["loss"] = loss_dict["total_recon_loss"] + loss_dict["total_vq_loss"]

        return loss_dict


class VQVAE_pl(pl.LightningModule):
    def __init__(self, args, criterion, encoder, vqvae_layer, atom_decoder, chiral_decoder, bond_decoder=None):
        super().__init__()
        self.args = args
        self.criterion = criterion
        self.model = GraphVQVAE(criterion, encoder, vqvae_layer, atom_decoder, chiral_decoder, bond_decoder)

        self._bnb_setting()

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)

        log_kwargs = {
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "sync_dist": True,
            "batch_size": len(batch),
        }
        for k, v in loss_dict.items():
            if k == "loss":
                self.log("train_loss", v.item(), prog_bar=True, **log_kwargs)
            else:
                self.log(k, v.item(), **log_kwargs)

        return loss_dict

    def configure_optimizers(self):
        lr = self.args.learning_rate
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.args.weight_decay)

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
