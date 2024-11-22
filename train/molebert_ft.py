import os
import wandb
import warnings
import os.path as osp
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

from utils import rank_zero_print
from args.molebert_ft_args import parser
from dataset import load_molebert_task_dataset
from models import GNN_graphpred, MoleBERTFT_pl

# for A6000: can be `medium` (bfloat16), `high` (tensorfloat32)
torch.set_float32_matmul_precision("high")

# cross-entropy loss fct (Eq. 5, L_mam)
criterion = nn.BCEWithLogitsLoss(reduction="none")


def init_wandb_and_get_run_id(args):
    @rank_zero_only
    def _init(args):
        run = wandb.init(
            name=args.version,
            dir=os.path.join(args.save_root, args.project),
            mode="offline" if not args.wandb else "online",
            project=args.project,
            id=f"{args.version}_{wandb.util.generate_id()}",
        )
        return run

    run = _init(args)

    if run is not None:
        run_time = datetime.fromtimestamp(run.start_time).strftime("%Y%m%d_%H%M%S")
        run_name = f"run-{run_time}-{run.id}"
        os.environ["WANDB_RUN_NAME"] = run_name

    run_name = os.environ.get("WANDB_RUN_NAME")
    assert run_name is not None, f"run_name is set `None`. retry."

    return run, run_name


def main():
    # arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.version = f"{data_args.dataset}-seed={data_args.split_seed}-{training_args.version}"
    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(",")]
    devices = training_args.gpus
    num_devices = len(devices)
    gpu_bs = training_args.per_device_train_batch_size
    desire_bs = training_args.desire_train_batch_size
    gradient_accumulation_steps = max(1, desire_bs // (gpu_bs * num_devices))
    if training_args.fp16:
        if training_args.bf16:
            warnings.warn("너 fp16 bf16 둘 다 줬어. 근데 fp16이 우선 순위니까 fp16 쓴다.")
        model_precision = "16"
    else:
        if training_args.bf16:
            model_precision = "bf16"
        else:
            model_precision = "32"

    # dataset
    num_tasks, dm = load_molebert_task_dataset(data_args, training_args)
    dm.setup()  # dataset을 로드하기 위해 setup 호출
    steps_per_epoch = len(dm.train_dataset) // ((training_args.per_device_train_batch_size) * num_devices)
    log_every_n_steps = min(50, steps_per_epoch // 5)  # epoch 당 최소 5번 로깅 보장

    # model
    gnn = GNN_graphpred(
        model_args.num_layer,
        model_args.emb_dim,
        num_tasks,
        JK=model_args.JK,
        drop_ratio=model_args.dropout_ratio,
        graph_pooling=model_args.graph_pooling,
        gnn_type=model_args.gnn_type,
    )
    gnn.from_pretrained(model_args.tmcl_ckpt_path)
    model = MoleBERTFT_pl(training_args, criterion, gnn, model_args.graph_pooling == "attention")

    # callbacks and strategy
    save_root = osp.join(training_args.save_root, training_args.project)
    run, run_name = init_wandb_and_get_run_id(training_args)
    ckpt_path = osp.join(save_root, "ckpt", run_name)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(osp.join(save_root, "wandb"), exist_ok=True)

    checkpoint_callback_train_loss = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="train_loss-{epoch:02d}-{train_loss:.4f}",
        monitor="train_loss",
        save_last=True,  # 마지막 epoch 무조건 저장 -> last.ckpt
        save_top_k=1,  # 가장 좋은 k개 유지. -1이면 모두 유지
        mode="min",
        every_n_epochs=1,  # 매 N 에폭마다 저장
        save_on_train_epoch_end=True,
    )

    checkpoint_callback_valid_auc = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="val_auc-{epoch:02d}-{val_auc:0.4f}",
        monitor="val_auc",
        save_last=False,  # 마지막 epoch 저장은 train_loss에서 함
        save_top_k=1,  # 가장 좋은 k개 유지. -1이면 모두 유지
        mode="max",
        every_n_epochs=1,  # 매 N 에폭마다 저장
        save_on_train_epoch_end=False,
    )

    wandb_logger = WandbLogger(experiment=run, version=run_name)

    if training_args.fdsp:
        rank_zero_print("===== Strategy: Fully Sharded Data Parallel =====")
        policy = {MoleBERTFT_pl}  # 분산시키고 싶은 모델의 모듈 (set)
        strategy = FSDPStrategy(
            auto_wrap_policy=policy,
            activation_checkpointing_policy=policy,
            # default values
            sharding_strategy="FULL_SHARD",
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        rank_zero_print("===== Strategy: DDP =====")
        strategy = "ddp"
        # strategy = DDPStrategy(find_unused_parameters=True)  # for debug

    # train
    trainer = Trainer(
        # w.r.t. HW
        accelerator="gpu",
        strategy=strategy,
        devices=num_devices,
        precision=model_precision,
        # w.r.t logging
        logger=wandb_logger,
        callbacks=[checkpoint_callback_train_loss, checkpoint_callback_valid_auc],
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=True,
        # w.r.t training
        max_epochs=training_args.num_train_epochs,
        accumulate_grad_batches=gradient_accumulation_steps,
    )
    trainer.fit(model, datamodule=dm)

    if hasattr(dm, "test_dataloader"):
        if dm.test_dataloader is not None:
            rank_zero_print("===== Test with Last ckpt =====")
            trainer.test(model, dm.test_dataloader(), ckpt_path=osp.join(ckpt_path, "last.ckpt"))

            best_train_loss_path = glob(osp.join(ckpt_path, "*train_loss*.ckpt"))[0]
            best_val_auc_path = glob(osp.join(ckpt_path, "*val_auc*.ckpt"))[0]

            rank_zero_print("===== Test with Best train loss ckpt =====")
            trainer.test(model, dm.test_dataloader(), ckpt_path=best_train_loss_path)

            rank_zero_print("===== Test with Best valid auc ckpt =====")
            trainer.test(model, dm.test_dataloader(), ckpt_path=best_val_auc_path)


if __name__ == "__main__":
    main()
