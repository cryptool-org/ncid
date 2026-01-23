import torch
from torch import nn
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()

        # saves parameters so that they can be saved and loaded later
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad
    def predict(self, input, batch_size):
        x = torch.tensor(input, dtype=torch.float32)

        outputs = []
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            out = self(batch)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)

        return F.softmax(outputs, dim=1)
