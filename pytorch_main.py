import torch
import codecs
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
torch.cuda.set_device(0)

LR = 0.001
EPOCH = 2
SEED = 0
with codecs.open("datasets/dialogue_nli/dialogue_nli_train.jsonl", "r", "utf8") as f:
    data = json.load(f)
'''
{'label': 'negative', 'sentence1': 'since my dad is a mechanic we had mostly car books .', 'sentence2': 'my dad is a priest .', 'triple1': ['my father', 'has_profession', 'mechanic'], 'triple2': ['my father', 'has_profession', 'priest'], 'dtype': 'e2swap_up', 'id': 'dialogue_nli_train_298308'}
{'label': 'positive', 'sentence1': 'i am only 22 so i would not know .', 'sentence2': 'i am twenty two years old .', 'triple1': ['i', 'has_age', '22'], 'triple2': ['i', 'has_age', '22'], 'dtype': 'matchingtriple_up', 'id': 'dialogue_nli_train_83252'}
{'label': 'neutral', 'sentence1': 'definitely ! ribeye is my favorite .', 'sentence2': 'i used to be enlisted in the army .', 'triple1': ['<none>', '<none>', '<none>'], 'triple2': ['i', 'previous_profession', 'army'], 'dtype': 'miscutterance_up', 'id': 'dialogue_nli_train_166990'}
'''
data = data[30000]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    torch.cuda.manual_seed(SEED)
else:
    torch.manual_seed(SEED)


class NliDataset(Dataset):
    def __init__(self, data_path):
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass


class NliClassifier(nn.Module):
    def __init__(self):
        super(NliClassifier, self).__init__()
        pass

    def forward(self):
        pass


def train_valid(model, train_dataloader, valid_dataloader, optimizer, criterion, multistep_schedule):
    model.train()
    for step, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad() # clear the history gradient
        output = model(x)
        loss = criterion(output, y)
        loss.backward() # compute the gradient
        optimizer.step() # update the parameters based on the gradient

    model.eval()
    for step, (x, y) in enumerate(valid_dataloader):
        with torch.no_grad():
            output = model(x)
        loss = criterion(output, y)
        # update the LR
    multistep_schedule.step()
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
    train_dataset = NliDataset("datasets/dialogue_nli/dialogue_nli_train.jsonl")
    test_dataset = NliDataset("datasets/dialogue_nli/dialogue_nli_test.jsonl")
    valid_dataset = NliDataset("datasets/dialogue_nli/dialogue_nli_dev.jsonl")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    model = NliClassifier()
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #  decay the LR when EPOCH//2 and EPOCH//4*3, the scope is 0.1
    multistep_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCH//2, EPOCH//4*3], gamma=0.1)
    for epoch in range(EPOCH):
        train_valid(model, train_dataloader, valid_dataset, optimizer, criterion, multistep_schedule)
    test("", test_dataloader)


if __name__ == '__main__':
    main()


