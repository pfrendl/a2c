import argparse
from pathlib import Path
from typing import Any

import cv2
import torch

from environments.base.step_limit_env import StepLimitEnv
from environments.blob_env import BlobEnv
from models.blob_model import BlobModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint_path", type=Path, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    args = parser.parse_args()

    checkpoint: dict[str, Any] = torch.load(args.checkpoint_path)
    device = torch.device(args.device)

    train_args = checkpoint["args"]

    env = StepLimitEnv(env=BlobEnv(), max_steps=1000)
    model = BlobModel(selector_size=train_args.selector_size, state_size=train_args.state_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_state = torch.zeros((1, train_args.state_size), device=device)
    initial_prev_action_idx = 0
    initial_prev_reward = 0.0

    position, num_objects, objects, progress = env.reset()
    state = initial_state
    prev_action_idx = initial_prev_action_idx
    prev_reward = initial_prev_reward

    while True:
        action_logits, state_value_pred, state, debug = model(
            position=torch.tensor(position, dtype=torch.float32, device=device)[None, :],
            num_objects=torch.tensor(num_objects, dtype=torch.int32, device=device)[None, :],
            objects=torch.tensor(objects, dtype=torch.float32, device=device)[None, :],
            progress=torch.tensor(progress, dtype=torch.float32, device=device)[None, :],
            prev_action_idx=torch.tensor([prev_action_idx], dtype=torch.int64, device=device),
            prev_reward=torch.tensor([[prev_reward]], dtype=torch.float32, device=device),
            state=state,
        )
        action_probs = torch.softmax(action_logits, dim=1)
        action_idx = torch.distributions.Categorical(probs=action_probs).sample().item()

        attention_weights = debug[0].cpu().detach().numpy()
        img = env.env.visualize(resolution=600, attention_weights=attention_weights)
        cv2.imshow("display", img)
        cv2.waitKey(1)

        [position, num_objects, objects, progress], reward, done, _ = env.step(action_idx)
        prev_action_idx = action_idx
        prev_reward = reward

        if done:
            position, num_objects, objects, progress = env.reset()
            state = initial_state
            prev_action_idx = initial_prev_action_idx
            prev_reward = initial_prev_reward
