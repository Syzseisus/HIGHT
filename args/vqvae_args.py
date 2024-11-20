from dataclasses import field, dataclass

import torch
from transformers import HfArgumentParser


# fmt: off
@dataclass
class TrainingArguments:
    # w.r.t. HW
    gpus: str = field(default="0,1,2,3", metadata={"help": "사용할 gpu번호 리스트"})
    num_workers: int = field(default=8, metadata={"help": "number of workers for dataset loading"})
    fp16: bool = field(default=False, metadata={"help": "우선 순위. 이거 주면 무조건 `Float16` 사용"})
    bf16: bool = field(default=False, metadata={"help": "차순위. 이걸 줘도 `fp16 = True` 있으면 `Float16` 사용. `fp16=False`고 `bf16=True`여야 `BFloat16` 사용."})
    fdsp: bool = field(default=False, metadata={"help": "모델이 너무 커서 bs=1도 안 될 때 `Fully Sharded Data Parallel`를 사용하기 위함. `policy`를 따로 설정해줘야된다."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # w.r.t. step
    num_train_epochs: int = field(default=60, metadata={"help": "num_epoch"})
    learning_rate: float = field(default=0.001, metadata={"help": "lr"})
    weight_decay: float = field(default=0, metadata={"help": "weight decay"})
    lr_scheduler_type: str = field(default="lambda", metadata={"help": "which lr scheduler to use", "choices": ["lambda", "linear", "cosine", "warmup_cosine"]})
    warmup_epochs: int = field(default=0, metadata={"help": "epoch to warmup before scheduling"})
    # `per_device_batch_size`에는 갖고 있는 GPU로 처리 가능한 사이즈를 입력하고
    # `desire_batch_size`에는 실제로 한 step에서 처리하고 싶은 배치 사이즈를 입력하면
    # 그에 맞도록 아래와 같이 accumulation 계산해서 처리해줌
    # gradient_accumulation_steps = max(1, desire_batch_size // (per_device_batch_size * num_devices))
    per_device_train_batch_size: int = field(default=64, metadata={"help": "GPU 별로 사용할 배치사이즈."})
    desire_train_batch_size: int = field(default=256, metadata={"help": "실제로 원하는 배치 사이즈."})
    # w.r.t. saving arguments
    """
    ./{save_root}
    └── {project}
        ├── "ckpt"
        │   ├── run-{{timestamp}}-{version}-{{random_id}}
                └── ~.ckpt
        └── "wandb"
            ├── latest-run
            ├── run-{{timestamp}}-{version}-{{random_id}}
                ├── files/
                │   ├── output.log
                │   ├── requirements.txt
                │   └── wandb-metadata.json
                ├── logs/
                │   ├── debug-core.log
                │   ├── debug-internal.log
                │   └── debug.log
                ├── tmp/
                │   └── code/
                │       └── // empty
                ├── output.log
                └── run-{version}-{{random_id}}.wandb
            ├── debug-internal.log
            └── debug.log
    - {}는 아래의 argument들임
    - {{}}는 `wandb.init`을 통해 추출한 값들임
    - wandb가 run-{{timestamp}}-{{random_id}}를 하드코딩 해둬서, `wandb.init` 하고 추출해서 `ckpt_path` 정의
    """
    save_root: str = field(default="/where/to/save/", metadata={"help": "어디에 저장할 건지"})
    project: str = field(default="HIGHT", metadata={"help": "wandb project 이름"})
    version: str = field(default="train_tokenizer", metadata={"help": "wandb version 이름"})
    wandb: bool = field(default=True, init=False, metadata={"help": "wandb를 online으로 쓸지말지 결정. `wandb_off`로만 변경 가능"})
    wandb_off: bool = field(default=False, metadata={"help": "이거 주면 offline에서 wandb 로깅"})

    def __post_init__(self):
        if self.wandb_off:
            self.wandb = False
        if torch.cuda.is_available() and self.gpus:
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    @property
    def device(self) -> "torch.device":
        """mimic transformers.TrainingArguments`"""
        return self._device

@dataclass
class ModelArguments:
    # w.r.t GNN
    num_layer: int = field(default=5, metadata={"help": "number of GNN message passing layers (default: 5)."})
    emb_dim: int = field(default=300, metadata={"help": "embedding dimensions (default: 300)"})
    num_tokens: int = field(default=640, metadata={"help": "number of atom tokens (Mole-BERT: 512)"})
    beta: float = field(default=0.25, metadata={"help": "beta"})
    edge: bool = field(default=True, metadata={"help": "whether to decode edges or not together with atoms"})
    dropout_ratio: float = field(default=0.0, metadata={"help": "dropout ratio"})
    JK: str = field(default="last", metadata={"help": "how the node features are combined across layers. last, sum, max or concat", "choices": ["last",]})
    gnn_type: str = field(default="gin", metadata={"help": "type of GNNs.", "choices": ["gin",]})
    decoder: str = field(default="gin", metadata={"help": "type of decoder layer", "choices": ["gin",]})


@dataclass
class DataArguments:
    data_path: str = field(default="/workspace/DATASET/MoleBERT/dataset", metadata={"help": "root directory of dataset for pretraining"})
    dataset: str = field(default="zinc_standard_agent", metadata={"help": "dataset name"})


parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
