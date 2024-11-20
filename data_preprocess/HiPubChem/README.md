1. PubChem 먼저 구축하기
    - smiles-text만 있으면 됨
2. BRICS로 찢기
    - refer: HiMol
3. positive / negative 만들기
    - positive pair: `"This molecule has <#> of <functional group name> groups."`
    - negative pair: `"This molecule has no <functional group name> groups."`
    - 추출된 `<functional group name>`이 없을 경우 `"This molecule has 0 functional groups."`로 통일
4. 예시는 `./sample.json` 파일 참조
    - `{index: {smiles: STR, has_group: STR, description: STR}}` 형태의 dictionary
    - smiles - description은 기존 PubChem에서 제공