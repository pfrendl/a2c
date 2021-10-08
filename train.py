import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch_optimizer import RAdam

from environments.base.environment import E
from environments.base.monitor_env import MonitorEnv
from environments.base.step_limit_env import StepLimitEnv
from environments.batch_env import BatchEnv
from environments.blob_env import BlobEnv
from models.blob_model import BlobModel


def train(envs: list[E], model: nn.Module, args: argparse.Namespace):
    state_size: int = args.state_size
    gamma: float = args.gamma
    tau: float = args.tau
    lr: float = args.lr
    update_interval: int = args.update_interval
    save_interval: int = args.save_interval
    frame_count: int = args.frame_count
    num_workers: int = args.num_workers
    device = torch.device(args.device)
    log_dir: Path = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=False)

    batch_env = BatchEnv(envs=envs)
    optimizer = RAdam(model.parameters(), lr=lr)
    initial_state = torch.zeros((num_workers, state_size), device=device)
    initial_prev_action_idx = torch.zeros((num_workers,), dtype=torch.int64, device=device)
    initial_prev_reward = torch.zeros((num_workers, 1), device=device)

    observation = batch_env.reset()
    observation = [o.to(device) for o in observation]
    state = initial_state
    prev_action_idx = initial_prev_action_idx
    prev_reward = initial_prev_reward

    for epoch in range(frame_count // (update_interval * num_workers)):

        if epoch % save_interval == 0:
            torch.save(
                {
                    "args": args,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                log_dir / f"checkpoint_{epoch:08d}.ckpt",
            )

        action_log_prob_arr = []
        entropy_arr = []
        state_value_pred_arr = []
        reward_arr = []
        ndone_arr = []

        for _ in range(update_interval):
            action_logits, state_value_pred, state, _ = model(
                *observation, prev_action_idx=prev_action_idx, prev_reward=prev_reward, state=state
            )

            action_probs = torch.softmax(action_logits, dim=1)
            action_log_probs = torch.log_softmax(action_logits, dim=1)
            entropy = -(action_probs * action_log_probs).sum(dim=1, keepdim=True)

            action_idx = torch.distributions.Categorical(probs=action_probs).sample()
            action_log_prob = action_log_probs[range(num_workers), action_idx][:, None]

            observation, reward, done = batch_env.step(action_idx=action_idx)
            observation = [o.to(device) for o in observation]
            reward = reward.to(device)
            done = done.to(device)

            state = torch.where(done, initial_state, state)
            prev_action_idx = torch.where(done.squeeze(-1), initial_prev_action_idx, action_idx)
            prev_reward = torch.where(done, initial_prev_reward, reward)

            action_log_prob_arr.append(action_log_prob)
            entropy_arr.append(entropy)
            state_value_pred_arr.append(state_value_pred)
            reward_arr.append(reward)
            ndone_arr.append(1.0 - done.to(dtype=torch.float32))

        state = state.detach()

        with torch.no_grad():
            _, G, _, _ = model(*observation, prev_action_idx=prev_action_idx, prev_reward=prev_reward, state=state)
        state_value_pred_arr.append(G)

        gae = torch.zeros((num_workers, 1), device=device)
        policy_loss = torch.zeros((num_workers, 1), device=device)
        value_loss = torch.zeros((num_workers, 1), device=device)
        for t in reversed(range(len(reward_arr))):
            G = reward_arr[t] + ndone_arr[t] * gamma * G

            # Generalized Advantage Estimataion
            value_target = reward_arr[t] + ndone_arr[t] * gamma * state_value_pred_arr[t + 1]
            td_error = (value_target - state_value_pred_arr[t]).detach()
            gae = td_error + ndone_arr[t] * gamma * tau * gae

            policy_loss -= gae * action_log_prob_arr[t] + 1e-2 * entropy_arr[t]
            value_loss += (G - state_value_pred_arr[t]) ** 2

        loss = (policy_loss + 0.5 * value_loss).mean(dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_env.stop()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selector-size", type=int, default=64, help="Model selector size")
    parser.add_argument("--state-size", type=int, default=128, help="Model state size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.9, help="Generalized Advantage Estimator discount factor")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--update-interval", type=int, default=5, help="Update interval")
    parser.add_argument("--save-interval", type=int, default=5000, help="Save interval")
    parser.add_argument("--frame-count", type=int, default=int(2e10), help="Number of frames to train for")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes (batch size)")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("output") / time.strftime("%Y%m%d-%H%M%S"), help="Log directory"
    )
    args = parser.parse_args()

    envs = [
        MonitorEnv(env=StepLimitEnv(env=BlobEnv(), max_steps=1000), log_dir=args.log_dir / f"env_{env_idx}")
        for env_idx, _ in enumerate(range(args.num_workers))
    ]
    model = BlobModel(selector_size=args.selector_size, state_size=args.state_size).to(args.device)
    train(envs, model, args)


if __name__ == "__main__":
    main()
