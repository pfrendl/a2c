import math

import torch
import torch.nn as nn
from torch import Tensor


class ObjectSelector(nn.Module):
    def __init__(self, hidden_size: int, context_size: int, object_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.fc0 = nn.Linear(object_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)

        self.key_value = nn.Linear(hidden_size, 2 * hidden_size)
        self.query = nn.Linear(context_size, hidden_size)

    def forward(self, objects_list: list[Tensor], context: Tensor) -> tuple[Tensor, list[Tensor]]:
        split_sections = [len(obj) for obj in objects_list]
        split_sections_ten = torch.tensor(split_sections, device=context.device)

        x = torch.cat(objects_list, dim=0)
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))

        key, value = self.key_value(x).chunk(2, dim=1)
        query = torch.repeat_interleave(self.query(context), split_sections_ten, dim=0)

        logits = torch.einsum("of,of->o", [query, key]) / math.sqrt(self.hidden_size)
        logits_list = logits.split(split_sections, dim=0)
        weights_list = [torch.softmax(logits, dim=0) for logits in logits_list]

        values_list = value.split(split_sections, dim=0)

        embeddings_list = [torch.einsum("o,of->f", [w, v]) for w, v in zip(weights_list, values_list)]
        embedding = torch.stack(embeddings_list, dim=0)

        return embedding, weights_list
