import math

import torch
import torch.nn as nn
from torch import optim
import lightning.pytorch as pl
from torchmetrics import MeanMetric, MetricCollection
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR, CosineAnnealingLR

from utils import rank_zero_print

NUM_BOND_ATTR = 4


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


class MoleBERT(nn.Module):
    def __init__(self, args, criterion, mnm_module, tokenizer, mask_edge=False):
        super().__init__()
        self.args = args
        self.mask_edge = mask_edge

        self.criterion = criterion
        self.mnm_module = mnm_module
        self.tokenizer = tokenizer
        self.node_lin_pred_head1 = nn.Linear(self.args.emb_dim, self.args.num_tokens)
        self.node_lin_pred_head2 = nn.Linear(self.args.emb_dim, self.args.num_tokens)
        if self.mask_edge:
            self.edge_lin_pred_head1 = nn.Linear(self.args.emb_dim, NUM_BOND_ATTR)
            self.edge_lin_pred_head2 = nn.Linear(self.args.emb_dim, NUM_BOND_ATTR)

    def forward(self, batch_dict):
        org = batch_dict["org"]
        mask1 = batch_dict["mask1"]
        mask2 = batch_dict["mask2"]

        # node_rep : for L_mam | graph_rep : for L_tri
        node_rep1, graph_rep1 = self.mnm_module.forward_cl(mask1.x, mask1.edge_index, mask1.edge_attr, mask1.batch)
        node_rep2, graph_rep2 = self.mnm_module.forward_cl(mask2.x, mask2.edge_index, mask2.edge_attr, mask2.batch)

        with torch.no_grad():
            _, graph_rep = self.mnm_module.forward_cl(org.x, org.edge_index, org.edge_attr, org.batch)

            # quantize -> make labels for L_mam
            atom_ids = self.tokenizer.get_codebook_indices(org.x, org.edge_index, org.edge_attr)
            node_labels1 = atom_ids[mask1.masked_atom_indices]
            node_labels2 = atom_ids[mask2.masked_atom_indices]
            edge_labels1 = mask1.mask_edge_label[:, 0]
            edge_labels2 = mask2.mask_edge_label[:, 0]

        # L_con (Eq. 7) | L_tri (Eq. 6, 7) | L_tmcl (Eq. 7)
        loss_cl = self.mnm_module.loss_cl(graph_rep1, graph_rep2)
        loss_tri = self.mnm_module.loss_tri(graph_rep, graph_rep1, graph_rep2)
        loss_tmcl = loss_cl + self.args.mu * loss_tri
        loss_dict = {
            "loss_cl": loss_cl,
            "loss_tri": loss_tri,
            "loss_tmcl": loss_tmcl,
        }

        # L_mam (Eq. 5)
        pred_node1 = self.node_lin_pred_head1(node_rep1[mask1.masked_atom_indices])
        pred_node2 = self.node_lin_pred_head2(node_rep2[mask2.masked_atom_indices])
        loss_mask = self.criterion(pred_node1.double(), node_labels1)
        loss_mask += self.criterion(pred_node2.double(), node_labels2)
        loss_mask_atom = loss_mask
        loss_dict["loss_atom"] = loss_mask_atom

        # compute accuracy
        acc_node1 = compute_accuracy(pred_node1, node_labels1)
        acc_node2 = compute_accuracy(pred_node2, node_labels2)
        acc_node = (acc_node1 + acc_node2) * 0.5
        acc_dict = {"node": acc_node}

        if self.mask_edge:
            # mask edge and extract representation
            masked_edge_index1 = mask1.edge_index[:, mask1.connected_edge_indices]
            edge_rep1 = node_rep1[masked_edge_index1[0]] + node_rep1[masked_edge_index1[1]]
            masked_edge_index2 = mask2.edge_index[:, mask2.connected_edge_indices]
            edge_rep2 = node_rep2[masked_edge_index2[0]] + node_rep2[masked_edge_index2[1]]

            # reconstruction loss (like L_mam)
            pred_edge1 = self.edge_lin_pred_head1(edge_rep1)
            pred_edge2 = self.edge_lin_pred_head2(edge_rep2)
            loss_mask_bond = self.criterion(pred_edge1.double(), edge_labels1)
            loss_mask_bond += self.criterion(pred_edge2.double(), edge_labels2)
            loss_mask += loss_mask_bond
            loss_dict["loss_bond"] = loss_mask_bond

            # compute accuracy
            acc_edge1 = compute_accuracy(pred_edge1, mask1.mask_edge_label[:, 0])
            acc_edge2 = compute_accuracy(pred_edge2, mask2.mask_edge_label[:, 0])
            acc_edge = (acc_edge1 + acc_edge2) * 0.5
            acc_dict["edge"] = acc_edge

        loss = loss_tmcl + loss_mask
        loss_dict["loss_mask"] = loss_mask
        loss_dict["loss"] = loss

        return loss_dict, acc_dict


class TMCL_pl(pl.LightningModule):
    def __init__(self, training_args, model_args, criterion, mnm_module, tokenizer, mask_edge):
        super().__init__()
        self.args = training_args
        self.mask_edge = mask_edge
        self.model = MoleBERT(model_args, criterion, mnm_module, tokenizer, mask_edge)

        metrics = {"node": MeanMetric()}
        if self.mask_edge:
            metrics["edge"] = MeanMetric()
        self.metrics = MetricCollection(metrics)

        self._bnb_setting()

    def training_step(self, batch, batch_idx):
        # forward
        batch_data = batch[0]
        loss_dict, acc_dict = self.model(batch_data)

        # loss
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

        # accuracy
        for k, v in acc_dict.items():
            self.metrics[k].update(v)
            self.log(k, self.metrics[k], **log_kwargs)

        return loss_dict

    def on_train_epoch_end(self):
        for metric in self.metrics.values():
            metric.reset()

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
