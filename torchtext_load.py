import json
import codecs
# v0.9
from torchtext.legacy.data import Field, Dataset, TabularDataset, BucketIterator, Iterator
from torchtext.vocab import GloVe
import torch
from torch.nn import init
from tqdm import tqdm
import torchtext.legacy.data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenize = lambda x: x.split()
sentence1 = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=100)
sentence2 = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=100)
corpus = Field(sequential=True, tokenize=tokenize, lower=True)
label = Field(sequential=False, use_vocab=False)
train_data, valid_data, test_data = TabularDataset.splits(
    path="datasets/dialogue_nli/processed/",
    train="dialogue_nli_train.jsonl",
    validation="dialogue_nli_dev.jsonl",
    test="dialogue_nli_test.jsonl",
    format="json",
    fields={'label': ('l', label),
            'sentence1': ('s1', sentence1),
            'sentence2': ('s2', sentence2),
            'corpus': ('crps', corpus)
            }
)
sentence1.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
sentence2.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
corpus.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
train_iteator, valid_iteator = BucketIterator.splits(
    (train_data, valid_data),
    batch_sizes=(32, 32),
    device=device,
    sort_key=lambda x: len(x.s1),
    sort_within_batch=False,
    repeat=False
)
test_iteator = Iterator(
    test_data,
    batch_size=32,
    device=device,
    train=False
)
for step, batch in enumerate(train_iteator):
    print(batch)
