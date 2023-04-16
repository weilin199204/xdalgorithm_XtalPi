# coding=utf-8

"""
Implementation of a SMILES dataset.
"""

import torch
import torch.utils.data as tud
import warnings


def filtering_illegal_tokens(smiles_list, legal_token_set, tokenizer):
    filtered_smiles_list = []
    for smiles in smiles_list:
        if 's+' in smiles:
            smiles = smiles.replace('[s+]', '[S+]')
        tokens = tokenizer.tokenize(smiles)
        illegal_tokens = list(set(tokens) - set(legal_token_set))
        if len(illegal_tokens) == 0:
            filtered_smiles_list.append(smiles)
        else:
            warnings.warn("SMILES {0} owning illegal tokens {1}, are ignored.".format(smiles,
                                                                                      ",".join(illegal_tokens)))
    return filtered_smiles_list


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, smiles_list, vocabulary, tokenizer):
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        legal_token_set = set([i for i in self._vocabulary._tokens.keys() if isinstance(i, str)])
        self._smiles_list = filtering_illegal_tokens(smiles_list, legal_token_set, self._tokenizer)

    def __getitem__(self, i):
        smi = self._smiles_list[i]
        if 's+' in smi:
            print(smi)
            smi = smi.replace('[s+]', '[S+]')
        tokens = self._tokenizer.tokenize(smi)
        encoded = self._vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._smiles_list)

    @staticmethod
    def collate_fn(encoded_seqs):
        """Converts a list of encoded sequences into a padded tensor"""
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


def calculate_nlls_from_model(model, smiles, batch_size=128):
    """
    Calculates NLL for a set of SMILES strings.
    :param model: Model object.
    :param smiles: List or iterator with all SMILES strings.
    :return : It returns an iterator with every batch.
    """
    dataset = Dataset(smiles, model.vocabulary, model.tokenizer)
    _dataloader = tud.DataLoader(dataset, batch_size=batch_size, collate_fn=Dataset.collate_fn)

    def _iterator(dataloader):
        for batch in dataloader:
            nlls = model.likelihood(batch.long())
            yield nlls.data.cpu().numpy()

    return _iterator(_dataloader), len(_dataloader)
