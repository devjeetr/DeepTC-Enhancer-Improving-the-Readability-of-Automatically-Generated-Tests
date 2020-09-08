from torch import is_tensor
from torch.utils.data import Dataset
import torch
from utilities.config import Config
from utilities.FileReader import FileReader
from utilities.preprocessing import context_encodings_to_tensors, preprocess_example
from utilities.preprocessing import pad_arr


class BPEDataSet(Dataset):
    def __init__(
        self,
        data_file,
        max_contexts,
        token_encoder,
        node_to_index,
        target_encoder,
        subtoken_len,
        ast_path_len,
        target_len,
        shuffle=True,
        variable_only_filter=True,
        line_cache=None,
    ):
        self.token_encoder = token_encoder
        self.node_to_index = node_to_index
        self.target_encoder = target_encoder
        self.subtoken_len = subtoken_len
        self.ast_path_len = ast_path_len
        self.target_len = target_len
        self.shuffle = shuffle
        self.max_contexts = max_contexts
        self.variable_only_filter = variable_only_filter
        self.reader = FileReader(data_file, line_cache=line_cache)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        label, contexts = preprocess_example(
            self.reader[idx],
            self.token_to_index,
            self.node_to_index,
            self.target_to_index,
            sos_idx=self.target_sos_idx,
            eos_idx=self.target_eos_idx,
            subtoken_len=self.subtoken_len,
            ast_path_len=self.ast_path_len,
            target_len=self.target_len,
            relevant_only=self.variable_only_filter
        )

        (
            start,
            end,
            path,
            masks,
            start_lengths,
            end_lengths,
            ast_lengths,
        ) = context_encodings_to_tensors(
            contexts,
            self.max_contexts,
            self.subtoken_len,
            self.ast_path_len,
            self.token_pad_idx,
            self.ast_pad_idx,
            shuffle=True,
        )

        label = pad_arr(label, self.target_len + 2, self.target_pad_idx)
        label = torch.tensor(label)

        return label, (start, end, path, masks, start_lengths, end_lengths, ast_lengths)

    def __len__(self):
        return len(self.reader)
