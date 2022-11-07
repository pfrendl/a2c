import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base.layers import Linear
from models.base.object_selector import ObjectSelector


class BlobModel(nn.Module):
    def __init__(self, selector_size: int, state_size: int):
        super().__init__()
        self.object_selector = ObjectSelector(hidden_size=selector_size, context_size=state_size, object_size=3)
        self.rnn = nn.GRUCell(input_size=2 + selector_size + 1 + 5 + 1, hidden_size=state_size)
        self.action_logits = Linear(state_size, 5)
        self.state_value_pred = Linear(state_size, 1)

    def forward(
        self,
        position: Tensor,
        num_objects: Tensor,
        objects: Tensor,
        progress: Tensor,
        prev_action_idx: Tensor,
        prev_reward: Tensor,
        state: tuple[Tensor, Tensor],
    ) -> tuple[list[Tensor], Tensor, tuple[Tensor, Tensor], list[Tensor]]:
        objects_list = [obj[:num_obj] for num_obj, obj in zip(num_objects, objects)]
        object_embedding, object_weights_list = self.object_selector(objects_list, state)

        prev_action = F.one_hot(prev_action_idx, num_classes=5).to(torch.float32)
        rnn_input = torch.cat([position, object_embedding, progress, prev_action, prev_reward], dim=1)
        state = self.rnn(rnn_input, state)

        action_logits = self.action_logits(state)
        state_value_pred = self.state_value_pred(state)

        return action_logits, state_value_pred, state, object_weights_list
