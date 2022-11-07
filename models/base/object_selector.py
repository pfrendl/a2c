import math

import torch
import torch.nn as nn
from torch import Tensor

from models.base.layers import Linear, ReLU


class ObjectSelector(nn.Module):
    def __init__(self, hidden_size: int, context_size: int, object_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.kv = nn.Sequential(
            Linear(object_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 2 * hidden_size),
        )
        self.q = nn.Sequential(
            nn.Linear(context_size, hidden_size),
        )

    def forward(self, objects_list: list[Tensor], context: Tensor) -> tuple[Tensor, list[Tensor]]:
        split_sections = [len(obj) for obj in objects_list]
        split_sections_ten = torch.tensor(split_sections, device=context.device)

        key, value = self.kv(torch.cat(objects_list, dim=0)).chunk(2, dim=1)
        query = torch.repeat_interleave(self.q(context), split_sections_ten, dim=0)

        logits = torch.einsum("of,of->o", [query, key]) / math.sqrt(self.hidden_size)
        weights_list = [torch.softmax(l, dim=0) for l in logits.split(split_sections, dim=0)]
        values_list = value.split(split_sections, dim=0)

        embeddings_list = [torch.einsum("o,of->f", [w, v]) for w, v in zip(weights_list, values_list)]
        embedding = torch.stack(embeddings_list, dim=0)

        return embedding, weights_list
