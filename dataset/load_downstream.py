from dataset import ClassificationDataModule

CLASSIFICATION_DATASET = {
    "tox21": 12,
    "toxcast": 617,
    "sider": 27,
    "clintox": 2,
    "muv": 17,
    "hiv": 1,
    "bbbp": 1,
    "bace": 1,
}
REGRESSION_DATASET = ["esol", "lipo", "malaria", "cep"]
DTA_DATATSET = ["davis", "kiba"]


def load_molebert_task_dataset(data_args, training_args):
    data_name = data_args.dataset.lower()
    if data_name in CLASSIFICATION_DATASET:
        num_tasks = CLASSIFICATION_DATASET[data_name]
        dm = ClassificationDataModule(data_args, training_args)
    elif data_name in REGRESSION_DATASET:
        pass
    elif data_name in DTA_DATATSET:
        pass

    return num_tasks, dm
