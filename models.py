import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModule, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)

    def hidden(self):
        return (
            torch.randn(1, 1, self.hidden_size),
            torch.randn(1, 1, self.hidden_size)
        )

    def forward(self, inputs, hidden=None):
        lstm_out, hidden = self.lstm(
            inputs.view(len(inputs), 1, -1),
            hidden or self.hidden()
        )
        output = self.output(lstm_out.view(len(inputs), -1))
        scores = F.log_softmax(output, dim=1)
        return scores, hidden


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModule, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)

    def hidden(self):
        return torch.rand(1, 1, self.hidden_size)

    def forward(self, inputs, hidden=None):
        rnn_out, hidden = self.rnn(
            inputs.view(len(inputs), 1, -1),
            hidden or self.hidden()
        )
        output = self.output(rnn_out.view(len(inputs), -1))
        scores = F.log_softmax(output, dim=1)
        return scores, hidden


def get_tensors(article, char_encoding, device):
    char2int, int2char, int2hot, str2hot, hot2int, hot2str = char_encoding
    x = str2hot(article[1:])
    y = [char2int[c] for c in article[:-1]]

    x = torch.tensor(x, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    return x, y


def train(net, articles, char_encoding):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(net.parameters())

    for article in articles:
        x, y = get_tensors(article, char_encoding, device)

        output, hidden = net(x)
        print(output, y)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict(net):
    with torch.no_grad():
        previous = initial_words[0]

        for char in initial_words:
            yield previous
            previous, hidden = net(previous, hidden)

        for _ in range(100):
            yield previous
            previous, hidden = net(previous, hidden)
