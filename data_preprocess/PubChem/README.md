* 이 폴더에서 PubChem의 SMILES-Text pair를 생성한다.

* 우선 임시로, MolCA에서 제공해준 pair를 사용한다.
    - `./molca_pt_data.py`로 `root` 폴더에 저장된 MolCA에서 제공한 `.pt` 파일을 `.json` 파일로 변환
    - `{index: {smiles: "SMILES", text: "TEXT"}}` 형태로 저장