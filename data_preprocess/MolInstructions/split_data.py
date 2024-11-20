import os.path as osp

from datasets import load_dataset

# 0. root 지정
data_root = "/workspace/DATASET/Mol-Instructions/data/Molecule-oriented_Instructions"


# 1. load dataset
dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", trust_remote_code=True)

# 2. split dataset
for task_name in ["property_prediction", "forward_reaction_prediction", "reagent_prediction", "retrosynthesis"]:
    print(f"Process {task_name} task dataset")
    train_data = dataset[task_name].filter(lambda x: eval(x["metadata"])["split"] == "train")
    test_data = dataset[task_name].filter(lambda x: eval(x["metadata"])["split"] == "test")
    train_data.to_json(osp.join(data_root, task_name, "train.json"))
    test_data.to_json(osp.join(data_root, task_name, "test.json"))
    print("done")

print("ALL DONE")
