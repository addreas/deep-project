import time
import numpy as np
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
            self.hidden() if hidden is None else hidden
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
        hidden = self.hidden() if hidden is None else hidden
        rnn_out, hidden = self.rnn(
            inputs.view(len(inputs), 1, -1),
            hidden
        )
        output = self.output(rnn_out.view(len(inputs), -1))
        scores = F.log_softmax(output, dim=1)
        return scores, hidden


def get_tensors(article, char_encoding, device):
    char2int, int2char, int2hot, str2hot, hot2int, hot2str = char_encoding
    x = str2hot(article[1:])
    # CrossEntropyLoss wants integer labels, not one hot vectors...
    y = [char2int[c] for c in article[:-1]]

    x = torch.tensor(x, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    return x, y


def progress(i, len_articles, times, message):
    print(f'Article {i}/{len_articles}.',
          f'Time per article: {np.mean(times):.2f}s',
          message,
          end='\r')


def train(net, articles, char_encoding):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(net.parameters())
    losses = []
    times = []
    try:
        for i, article in enumerate(articles):
            start_time = time.time()
            progress(i, len(articles), times, 'getting tensors......')
            x, y = get_tensors(article, char_encoding, device)

            progress(i, len(articles), times, 'forward pass.........')
            output, hidden = net(x)

            progress(i, len(articles), times, 'calculating loss.....')
            loss = criterion(output, y)
            losses.append(loss.item())

            optimizer.zero_grad()
            progress(i, len(articles), times, 'backward pass........')
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)

            progress(i, len(articles), times, 'updating parameters...')
            optimizer.step()

            step_time = time.time() - start_time
            times = [step_time, *times[:4]]

    except KeyboardInterrupt:
        progress(i, len(articles), times, 'Traing interrupted....')
        return losses

    return losses


def predict(net, char_encoding, initial_words="So there I was."):
    str2hot = char_encoding[3]
    hot2str = char_encoding[4]
    initial_hot = torch.tensor(str2hot(initial_words), dtype=torch.float)

    with torch.no_grad():  # probably quicker if we don't calculate gradients.
        previous, hidden = net(initial_hot)
        previous = previous[-1:, :]

        for _ in range(100):
            previous, hidden = net(previous, hidden)
            yield previous
