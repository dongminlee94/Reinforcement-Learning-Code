# Reinforcement Learning Code

## Papers

- [Deep Q-Network (DQN)](https://daiwk.github.io/assets/dqn.pdf)
- [Double DQN (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Advantage Actor-Critic (A2C)](http://incompleteideas.net/book/RLbook2018.pdf)
- [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Truncated Natural Policy Gradient (TNPG)](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Generalized Advantage Estimator (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Apprenticeship Learning via Inverse Reinforcement Learning (APP)](http://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Abbeel+Ng:2004.pdf)
- [Maximum Entropy Inverse Reinforcement Learning (MaxEnt)](http://new.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
- [Generative Adversarial Imitation Learning (GAIL)](https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf)
- [Variational Adversarial Imitation Learning (VAIL)](https://arxiv.org/pdf/1810.00821.pdf)

## Algorithms

### 01. Model-Free Reinforcement Learning

#### Deep Q-Network (DQN)

- [CartPole(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/cartpole/dqn)

#### Double DQN (DDQN)

- [CartPole(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/cartpole/ddqn)

#### Advantage Actor-Critic (A2C)

- [CartPole(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/cartpole/a2c)

#### Asynchronous Advantage Actor-Critic (A3C)

- [CartPole(Classic control)]()

#### Deep Deterministic Policy Gradient (DDPG)

- [Pendulum(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/pendulum/ddpg)

#### Truncated Natural Policy Gradient (TNPG)

- [Pendulum(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/pendulum/tnpg)
- [Hopper(MoJoCo)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mujoco/tnpg)

#### Trust Region Policy Optimization (TRPO)

- [Pendulum(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/pendulum/trpo)

#### TRPO + Generalized Advantage Estimator (GAE)

- [Pendulum(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/pendulum/trpo_gae)
- [Hopper(MoJoCo)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mujoco/trpo)

#### GAE + Proximal Policy Optimization (PPO)

- [Pendulum(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/pendulum/ppo_gae)
- [Hopper(MoJoCo)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mujoco/ppo)

#### Soft Actor-Critic (SAC)

- [Pendulum(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/pendulum/sac)
- [Hopper(MoJoCo)]()

---

### 02. Inverse Reinforcement Learning

#### Apprenticeship Learning via Inverse Reinforcement Learning (APP)

- [MountainCar(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mountaincar/app)

#### Maximum Entropy Inverse Reinforcement Learning (MaxEnt)

- [MountainCar(Classic control)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mountaincar/maxent)

#### Generative Adversarial Imitation Learning (GAIL)

- [Hopper(MoJoCo)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mujoco/gail)

#### Variational Adversarial Imitation Learning (VAIL)

- [Hopper(MoJoCo)](https://github.com/dongminleeai/Reinforcement-Learning-Code/tree/master/mujoco/vail)

---

## Learning curve

### CartPole

<img src="img/cartpole.png" width="500"/>

### Pendulum

<img src="img/pendulum.png" width="500"/>

### Hopper

---

## Reference

- [Minimal and Clean Reinforcement Learning Examples in PyTorch](https://github.com/reinforcement-learning-kr/reinforcement-learning-pytorch)
- [Pytorch implementation for Policy Gradient algorithms (REINFORCE, NPG, TRPO, PPO)](https://github.com/reinforcement-learning-kr/pg_travel)
- [Pytorch implementation of DDPG](https://github.com/jcwleo/Reinforcement_Learning/blob/master/pendulum/pendulum_ddpg.py)
- [Implementation of APP](https://github.com/jangirrishabh/toyCarIRL)
- [Implementation of MaxEnt](https://github.com/MatthewJA/Inverse-Reinforcement-Learning)
- [Pytorch implementation of GAIL](https://github.com/Khrylx/PyTorch-RL)
- [Pytorch implementation of SAC1](https://github.com/vitchyr/rlkit/tree/master/rlkit/torch/sac)
- [Pytorch implementation of SAC2](https://github.com/pranz24/pytorch-soft-actor-critic)
