## Steps required to go from raw source files -> data ready for training/inference

1. Process raw source files to create C2S dataset using JPredict (from original code2seq repo)

2. Conduct masking/extraction of variable paths from raw dataset:
    1. For **variable name prediction**, extract variable paths from raw C2S dataset using `preprocessing/create_var_name_dataset`
    2. For **method name prediction**, mask variables from all context paths using `preprocessing/mask_identifiers`.

3. Create vocabularies:
    1. Create token frequency maps using `preprocessing/construct_counts`
    2. Create SentencePiece vocabulary for subtokes and targets using `utilities/train_bpe`
    3. Create `node_dict.pkl` using construct node\_dict.


