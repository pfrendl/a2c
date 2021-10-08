# A2C
An implementation of the `Synchronous Advantage Actor Critic (A2C)` reinforcement learning algorithm in PyTorch.

Simplicty, clarity and modularity are top priorities. The environments are run in child processes for extra performance. The code uses Python 3.9 features.

The current implementation is also an experiment on an architectural prior: the observation contains multiple attribute vectors, each representing an object surrounding the agent. The agent behavior is invariant to the ordering of said vectors. The prior is implemented using an attention mechanism.

The agent has to avoid some objects and touch others to collect them and earn rewards.
![image](https://user-images.githubusercontent.com/6968154/136024997-370bc4ce-335e-4603-a80d-04122bdb7a9a.png)

https://user-images.githubusercontent.com/6968154/136035538-e363014a-459d-4e37-bc32-079ed2c2bfb7.mp4

The training is integrated with Tensorboard:
![image](https://user-images.githubusercontent.com/6968154/136025860-dd18469e-e439-45ac-810b-19d79e0dee6a.png)
