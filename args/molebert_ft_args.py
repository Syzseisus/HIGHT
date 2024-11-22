from dataclasses import field, dataclass

import torch
from transformers import HfArgumentParser

from models import GNN_graphpred
from dataset import DTA_DATATSET, REGRESSION_DATASET, CLASSIFICATION_DATASET

"""
tmcl_args.py와 거의 비슷
1. `TrainingArguments.lr_scale`: NEW
2. `TrainingArgument.per_device_train_batch_size` -> valid, test 추가
3. `TrainingArgument.desire_train_batch_size` -> valid, test 추가
4. `ModelArguments.dropout_ratio`: 0.0 -> 0.5
5. `ModelArguments.graph_pooling`: NEW
6. `DataArguments.dataset`: zinc_standard_agent -> check
    - 이를 위해 `CLASSIFICATION_DATASET`, `REGRESSION_DATASET`, `DTA_DATATSET`을 dataset에서 가져옴
7. `ModelArguments.vqvae_ckpt_path` -> `tmcl_ckpt_path`
8. `DataArguments.split_seed`: NEW
9. `DataArguments.split`: NEW
10. `DataArguments.mask_*`: DELETE
"""


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
    num_train_epochs: int = field(default=100, metadata={"help": "num_epoch"})
    learning_rate: float = field(default=0.001, metadata={"help": "lr"})
    lr_scale: float = field(default=1.0, metadata={"help": "relative learning rate for the feature extraction layer (default: 1)"})
    weight_decay: float = field(default=0, metadata={"help": "weight decay"})
    lr_scheduler_type: str = field(default="", metadata={"help": "which lr scheduler to use", "choices": ["", "lambda", "linear", "cosine", "warmup_cosine"]})
    warmup_epochs: int = field(default=0, metadata={"help": "epoch to warmup before scheduling"})
    # `per_device_batch_size`에는 갖고 있는 GPU로 처리 가능한 사이즈를 입력하고
    # `desire_batch_size`에는 실제로 한 step에서 처리하고 싶은 배치 사이즈를 입력하면
    # 그에 맞도록 아래와 같이 accumulation 계산해서 처리해줌
    # gradient_accumulation_steps = max(1, desire_batch_size // (per_device_batch_size * num_devices))
    per_device_train_batch_size: int = field(default=64, metadata={"help": "GPU 별로 사용할 배치사이즈."})
    desire_train_batch_size: int = field(default=256, metadata={"help": "실제로 원하는 배치 사이즈."})
    per_device_valid_batch_size: int = field(default=64, metadata={"help": "GPU 별로 사용할 배치사이즈."})
    desire_valid_batch_size: int = field(default=256, metadata={"help": "실제로 원하는 배치 사이즈."})
    per_device_test_batch_size: int = field(default=64, metadata={"help": "GPU 별로 사용할 배치사이즈."})
    desire_test_batch_size: int = field(default=256, metadata={"help": "실제로 원하는 배치 사이즈."})
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
    mu: float = field(default=0.1, metadata={"help": "mu"})
    dropout_ratio: float = field(default=0.5, metadata={"help": "dropout ratio"})
    JK: str = field(default="last", metadata={"help": "how the node features are combined across layers. last, sum, max or concat", "choices": ["last",]})
    graph_pooling: str = field(default="mean", metadata={"help": "graph level pooling", "choices": ["sum", "mean", "max", "set2set2", "attention"]})
    gnn_type: str = field(default="gin", metadata={"help": "type of GNNs.", "choices": ["gin",]})
    decoder: str = field(default="gin", metadata={"help": "type of decoder layer", "choices": ["gin",]})
    tmcl_ckpt_path: str = field(default=None, metadata={"help": "tmcl.py를 통해 얻은 ckpt 경로. 여기서 `mnmmodule.gnn` 갖다 쓸거임"})

    def __post__init__(self):
        if not self.tmcl_ckpt_path:
            raise ValueError("`tmcl_ckpt_path`는 반드시 있어야 함")
        else:
            try:
                gnn = GNN_graphpred(5, 300, 1, "last", 0.5, "mean", "gin")
                gnn.from_pretrained(self.tmcl_ckpt_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"{self.tmcl_ckpt_path}라는 파일 없어")
            except RuntimeError as e:
                raise RuntimeError("`load_state_dict` 실패\n" + str(e))
            except Exception as e:
                raise Exception("알 수 없는 에러 발생:\n" + str(e))


@dataclass
class DataArguments:
    data_path: str = field(default="/workspace/DATASET/MoleBERT/dataset", metadata={"help": "root directory of dataset for pretraining"})
    dataset: str = field(default="check", metadata={"help": "dataset name", "choices": ["check"] + list(CLASSIFICATION_DATASET.keys()) + REGRESSION_DATASET + DTA_DATATSET})
    split_seed: int = field(default=0, metadata={"help": "데이터셋 split할 때 사용할 seed."})
    split: str = field(default="scaffold", metadata={"help": "데이터셋 split 방법", "choices": ["random", "scaffold", "random_scaffold"]})

    def __post__init__(self):
        if self.dataset == "check":
            raise ValueError("`dataset` should be passed.")


parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
