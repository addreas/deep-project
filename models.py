import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1):
        super(LSTMModule, self).__init__()

        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(input_size, hidden_size, layers)
        self.output = nn.Linear(hidden_size, input_size)

    def hidden(self):
        return (
            torch.randn(self.layers, 1, self.hidden_size),
            torch.randn(self.layers, 1, self.hidden_size)
        )

    def forward(self, inputs, hidden=None):
        lstm_out, hidden = self.lstm(
            inputs.view(len(inputs), 1, -1),
            self.hidden() if hidden is None else hidden
        )
        output = self.output(lstm_out.view(len(inputs), -1))
        return output, tuple(h.detach() for h in hidden)


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1):
        super(RNNModule, self).__init__()

        self.hidden_size = hidden_size
        self.layers = layers

        self.rnn = nn.RNN(input_size, hidden_size, layers)
        self.output = nn.Linear(hidden_size, input_size)

    def hidden(self):
        return torch.rand(self.layers, 1, self.hidden_size)

    def forward(self, inputs, hidden=None):
        hidden = self.hidden() if hidden is None else hidden
        rnn_out, hidden = self.rnn(
            inputs.view(len(inputs), 1, -1),
            hidden
        )
        output = self.output(rnn_out.view(len(inputs), -1))
        return output, hidden.detach()


def get_hot_article(article, char_encoding, device):
    char2int, int2char, int2hot, str2hot, hot2int, hot2str = char_encoding
    x = str2hot(article[:-1])
    # CrossEntropyLoss wants integer labels, not one hot vectors...
    y = [char2int[c] for c in article[1:]]

    x = torch.tensor(x, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    return x, y


def get_batches(x, y, batch_size):
    num, _ = x.shape
    for batch in range(0, num - batch_size, batch_size):
        batch_end = batch + batch_size
        yield x[batch:batch_end, :], y[batch:batch_end]


def progress(i, len_articles, times, message):
    print(f'Article {i}/{len_articles}.',
          f'Last 5 mean time: {np.mean(times):.2f}s',
          message,
          end='\r')


def train(net, articles, char_encoding, batch_size=25, reset_hidden=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters())
    losses = []
    times = []
    try:
        hidden = None
        for i, article in enumerate(articles):
            start_time = time.time()

            progress(i, len(articles), times, 'getting tensors......')
            x, y = get_hot_article(article, char_encoding, device)

            if reset_hidden:
                hidden = None

            article_losses = []

            for x_batch, y_batch in get_batches(x, y, batch_size):

                progress(i, len(articles), times, 'forward pass.........')
                output, hidden = net(x_batch, hidden)

                progress(i, len(articles), times, 'calculating loss.....')
                loss = criterion(output, y_batch)
                article_losses.append(loss.item())

                optimizer.zero_grad()
                progress(i, len(articles), times, 'backward pass........')
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5)

                progress(i, len(articles), times, 'updating parameters..')
                optimizer.step()

            losses.append(np.mean(article_losses))
            step_time = time.time() - start_time
            times = [step_time, *times[:4]]

    except KeyboardInterrupt:
        progress(i, len(articles), times, 'Traing interrupted....')
        return losses

    return losses


def random_choice(out):
    out = out.flatten()
    out = F.softmax(out, dim=0)
    return np.random.choice(len(out), p=out.numpy())


def predict(net, encoding, initial_words, length=100):
    char_generator = predict_chars(net, encoding, initial_words, length=100)
    return "".join(c for c in char_generator)


def predict_chars(net, encoding, initial_words, length):
    int2char = encoding[1]
    int2hot = encoding[2]
    char2int, int2char, int2hot, str2hot, hot2int, hot2str = encoding
    initial_hot = torch.tensor(str2hot(initial_words), dtype=torch.float)

    yield from initial_words

    with torch.no_grad():  # probably quicker if we don't calculate gradients.
        out, hidden = net(initial_hot)
        next_int = random_choice(out[-1])

        for _ in range(length):
            yield int2char[next_int]
            next_hot = torch.tensor(int2hot[next_int], dtype=torch.float)

            out, hidden = net(next_hot.view(1, -1), hidden)
            next_int = random_choice(out)
