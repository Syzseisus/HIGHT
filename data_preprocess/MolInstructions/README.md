[Mol-Instructions](https://huggingface.co/datasets/zjunlp/Mol-Instructions)에서 제공하는 데이터셋. 

### 로드
* `dataset = load_dataset("zjunlp/Mol-Instructions", task_name, trust_remote_code=True)`으로 로드 할 수 있음.
    * `task_name = "Molecule-oriented_Instructions"`의 경우, 여섯 개의 task를 key로 갖고 있는 `datasets.DatasetDict` 객체


### 태스크
1. 5.3절의 "Moleculear Property Prediction"은 `property_prediction` task에 해당한다.
2. 5.5절의 "Chemical Reaction Prediction"은 `forward_reaction_prediction`, `reagent_prediction`, `retrosynthesis`의 세 가지 task에 해당한다.
3. 각 `dataset[task_name]`으로 추출된 `datasets.Dataset` 객체는 `instruction`, `input`, `output`, `metadata`를 키로 갖는 데이터를 포함한다.
4. `metadata`는 str으로 표현된, `task`와 `split`을 키로 갖는 dictionary이다.
5. 즉, `eval(data["metadata"])["task"]`를 통해 해당 데이터가 어떤 split에 속하는지 알 수 있다.


### 데이터 저장 위치
* `./split_data.py`를 실행하면 `.data_root/task_name/{train or test}.json`으로 저장된다.