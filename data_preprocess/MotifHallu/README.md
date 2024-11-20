* RDKit에서 제공하는 [38 개의 기본 functional group](https://github.com/rdkit/rdkit/blob/master/Data/FunctionalGroups.txt)의 유무를 활용해서 생성 (아래 표 참조)
* HiPubChem 데이터셋을 생성할 때 같이 생성됨
* `{index: {smiles: STR, instruction: STR, answer: STR}}` 형태의 dictionary
* 예시는 `./sample.json` 확인


* HIGHT Table 13. List of functional groups from RDKit used to construct MotifHallu. The functional group with the name "???" is neglected.
  |Chemical Representation|SMARTS|Name|
  |---|---|---|
  TODO