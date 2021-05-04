# Based on torch.utils.data Dataset, Dataloader
import torch
import codecs
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
#torch.cuda.set_device(0)
#torch.backends.cudnn.enabled=False

LR = 0.001
EPOCH = 2
SEED = 0
TRAIN_DATA_LEN = 30000
TRAIN_BATCH = 32
VALID_BATCH = 32
VALID_DATA_LEN = 10000
MAX_LEN = 50
W_EMBEDDING_DIM = 128
HIDDEN_DIM = 256
with codecs.open("datasets/dialogue_nli/dialogue_nli_train.jsonl", "r", "utf8") as f:
    train_data = json.load(f)

with codecs.open("datasets/dialogue_nli/dialogue_nli_dev.jsonl", "r", "utf8") as f:
    valid_data = json.load(f)
'''
with codecs.open("datasets/dialogue_nli/dialogue_nli_test.jsonl", "r", "utf8") as f:
    test_data = json.load(f)
{'label': 'negative', 'sentence1': 'since my dad is a mechanic we had mostly car books .', 'sentence2': 'my dad is a priest .', 'triple1': ['my father', 'has_profession', 'mechanic'], 'triple2': ['my father', 'has_profession', 'priest'], 'dtype': 'e2swap_up', 'id': 'dialogue_nli_train_298308'}
{'label': 'positive', 'sentence1': 'i am only 22 so i would not know .', 'sentence2': 'i am twenty two years old .', 'triple1': ['i', 'has_age', '22'], 'triple2': ['i', 'has_age', '22'], 'dtype': 'matchingtriple_up', 'id': 'dialogue_nli_train_83252'}
{'label': 'neutral', 'sentence1': 'definitely ! ribeye is my favorite .', 'sentence2': 'i used to be enlisted in the army .', 'triple1': ['<none>', '<none>', '<none>'], 'triple2': ['i', 'previous_profession', 'army'], 'dtype': 'miscutterance_up', 'id': 'dialogue_nli_train_166990'}
'''
train_data = train_data[:TRAIN_DATA_LEN]
valid_data = valid_data[:VALID_DATA_LEN]
corpus = [(sample["sentence1"].strip() + " " + sample["sentence2"].strip()) for sample in train_data]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    torch.cuda.manual_seed(SEED)
else:
    torch.manual_seed(SEED)


class Vocab:

    def __init__(self, p_corpus):
        self.corpus = p_corpus

        def build_wordlist(sentlist):
            wordlist = ['unk', '<pad>']
            for sentence in sentlist:
                wordlist.extend(sentence.split())
            return set(wordlist)

        def build_wordic(wordlist):
            word2id = {'<pad>': 0, '<unk>': 1}
            id2word = {0: '<pad>', 1: '<unk>'}
            for idx, word in enumerate(wordlist, 2):
                word2id.update({word: int(idx)})
                id2word.update({int(idx): word})
            return word2id, id2word
        self.wordset = build_wordlist(self.corpus)
        self.wrod2id, self.id2word = build_wordic(self.wordset)

    def string2ids(self, content):
        '''
        ids = []
        for word in content:
            if word in self.wrodset:
                ids.append(self.wrod2id[word])
            else:
                ids.append(self.wrod2id['<unk>'])
        return ids
        '''
        return [self.wrod2id[word] if word in self.wordset else self.wrod2id['<unk>'] for word in content.split()]

    def ids2string(self, ids):
        pass

    def __len__(self):
        return len(self.wordset)


class NliDataset(Dataset):

    def __init__(self, data, vocab):
        self.len = len(data)

        def padding_lab2num(unpad_data):
            for sample in unpad_data:
                if sample["label"] == "negative": sample["label"] = 0
                elif sample["label"] == "neutral": sample["label"] = 1
                else: sample["label"] = 2
                if len(sample["sentence1"].split()) < MAX_LEN:
                    tmp_list = []
                    pad_len = 0
                    pad_len = MAX_LEN - len(sample["sentence1"].split())
                    tmp_list = sample["sentence1"].split()
                    tmp_list.extend(["<pad>"] * pad_len)
                    sample["sentence1"] = " ".join(tmp_list)
                elif len(sample["sentence1"].split()) > MAX_LEN:
                    sample["sentence1"] = sample["sentence1"].split()[:MAX_LEN]
                if len(sample["sentence2"].split()) < MAX_LEN:
                    tmp_list = []
                    pad_len = 0
                    pad_len = MAX_LEN - len(sample["sentence2"].split())
                    tmp_list = sample["sentence2"].split()
                    tmp_list.extend(["<pad>"] * pad_len)
                    sample["sentence2"] = " ".join(tmp_list)
                elif len(sample["sentence2"].split()) > MAX_LEN:
                    sample["sentence2"] = sample["sentence2"].split()[:MAX_LEN]
            pad_data = unpad_data
            return pad_data
        data = padding_lab2num(data)
        for sample in data:
            sample["sentence1"] = vocab.string2ids(sample["sentence1"])
            sample["sentence2"] = vocab.string2ids(sample["sentence2"])
        self.data = data


    def __getitem__(self, index):

        # return (self.data[index]["label"],
        #         self.data[index]["sentence1"],
        #         self.data[index]["sentence2"])
        return (self.data[index]["label"], torch.IntTensor(np.asarray(self.data[index]["sentence1"])), torch.IntTensor(np.asarray(self.data[index]["sentence2"])))

    def __len__(self):
        return self.len


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
        self.activation = nn.ReLU()
        self.output = nn.Linear(HIDDEN_DIM, 3)

    def forward(self, encoded_s1, encoded_s2):
        concact_encoded = torch.cat([encoded_s1, encoded_s2], dim=1)
        return self.output(self.activation(self.hidden(concact_encoded)))


class NliClassifier(nn.Module):
    def __init__(self, vocab):
        super(NliClassifier, self).__init__()
        self.Embedding = nn.Embedding(len(vocab), W_EMBEDDING_DIM)
        self.LSTM = nn.LSTM(W_EMBEDDING_DIM, HIDDEN_DIM, num_layers=1, dropout=0)
        self.MLP = MLP().cuda()

        pass

    def forward(self, s1, s2):
        embedd_s1, embedd_s2 = self.Embedding(s1), self.Embedding(s2)
        torch.backends.cudnn.enabled = False
        encoded_s1, encoded_s2 = self.LSTM(embedd_s1.permute([1, 0, 2]))[0][-1], self.LSTM(embedd_s2.permute([1, 0, 2]))[0][-1]
        return self.MLP(encoded_s1, encoded_s2)


def train_valid(model, train_dataloader, valid_dataloader, optimizer, criterion, multistep_schedule, epoch):
    model.train()
    train_loss = []
    for step, (l, s1, s2) in enumerate(train_dataloader):
        optimizer.zero_grad() # clear the history gradient
        l, s1, s2 = l.cuda(), s1.cuda(), s2.cuda()

        output = model(s1,s2)
        loss = criterion(output, l)
        train_loss.append(loss)
        print("Epoch:{}/{} Training step:{} Loss:{:.4f}".format(epoch, EPOCH, step, loss.item()))
        loss.backward() # compute the gradient
        optimizer.step() # update the parameters based on the gradient
    train_loss_mean = train_loss / len(train_dataloader)
    model.eval()
    valid_loss = []
    acc = []
    correct = 0
    for step, (l, s1, s2) in enumerate(valid_dataloader):
        l, s1, s2 = l.cuda(), s1.cuda(), s2.cuda()
        with torch.no_grad():
            output = model(s1,s2)
        loss = criterion(output, l)
        print("Epoch:{}/{} Training step:{} Loss:{:.4f} Acc:{:.4f}".format(
            epoch, EPOCH, step, loss.item(), 100. * int((torch.argmax(output, dim=1) == l).sum())/VALID_BATCH))
        valid_loss.append(loss)
        acc.append(int((torch.argmax(output, dim=1) == l).sum())/VALID_BATCH)
        # update the LR
    multistep_schedule.step()
    valid_loss_mean = valid_loss / len(valid_dataloader)
    acc_mean = acc / len(valid_dataloader)
    print("Epoch:{}/{} Train Loss:{:.4f} Valid Loss:{:.4f} Acc:{:.4f}".format(
        epoch, EPOCH, train_loss_mean, valid_loss_mean, 100. * acc_mean))
    # save the model based on valid loss


def test(model_path, test_dataloader):
    best_model = torch.load(model_path).get('model').cuda()
    criterion = torch.load().get('criterion').cuda()
    best_model.eval()
    for step, (x, y) in enumerate(test_dataloader):
        with torch.no_grad():
            output = best_model(x)
        loss = criterion(output, y)
        final_predicted = torch.max(output, dim=1)
        # compute the acc


def main():
    vocab = Vocab(corpus)
    vocab_len = len(vocab)
    train_dataset = NliDataset(train_data, vocab)
    #test_dataset = NliDataset("datasets/dialogue_nli/dialogue_nli_test.jsonl")
    valid_dataset = NliDataset(valid_data, vocab)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=TRAIN_BATCH,
                                  shuffle=True
                                  )
    valid_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=VALID_DATA_LEN,
                                  shuffle=True
                                  )
    #test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    model = NliClassifier(vocab)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #  decay the LR when EPOCH//2 and EPOCH//4*3, the scope is 0.1
    multistep_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCH//2, EPOCH//4*3], gamma=0.1)
    for epoch in range(EPOCH):
        train_valid(model, train_dataloader, valid_dataloader, optimizer, criterion, multistep_schedule, epoch)
    #test("", test_dataloader)


if __name__ == '__main__':
    main()


