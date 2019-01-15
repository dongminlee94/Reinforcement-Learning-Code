import numpy as np
import torch
import torch.nn.functional as F
from utils import log_prob_density


def train_model(actor, critic, transition, actor_optim, critic_optim, args):
    states = torch.stack(transition.state)
    actions = torch.Tensor(transition.action).unsqueeze(1)
    rewards = torch.Tensor(transition.reward)
    masks = torch.Tensor(transition.mask)

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values)
    
    policies = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), policies)

#     criterion = torch.nn.MSELoss()
#     n = len(states)
#     arr = np.arange(n)
    
#     # ----------------------------
#     # step 2: get value loss and actor loss and update actor & critic
#     # batch를 random suffling하고 mini batch를 추출
#     for _ in range(10):
#         np.random.shuffle(arr)
        
#         for i in range(n // hp.batch_size): # batch_size = 64
#             batch_index = arr[hp.batch_size * i : hp.batch_size * (i + 1)]
#             batch_index = torch.LongTensor(batch_index)
            
#             inputs = torch.Tensor(states)[batch_index]
#             actions_samples = torch.Tensor(actions)[batch_index]
#             returns_samples = returns.unsqueeze(1)[batch_index]
#             advants_samples = advants.unsqueeze(1)[batch_index]
#             oldvalue_samples = old_values[batch_index].detach()
            
#             values = critic(inputs)
#             # clipping을 사용하여 critic loss 구하기 
#             clipped_values = oldvalue_samples + \
#                              torch.clamp(values - oldvalue_samples,
#                                          -hp.clip_param, # 0.2
#                                          hp.clip_param)
#             critic_loss1 = criterion(clipped_values, returns_samples)
#             critic_loss2 = criterion(values, returns_samples)
#             critic_loss = torch.max(critic_loss1, critic_loss2).mean()

#             # 논문에서 수식 6. surrogate loss 구하기
#             loss, ratio = surrogate_loss(actor, advants_samples, inputs,
#                                          old_policy.detach(), actions_samples,
#                                          batch_index)

#             # 논문에서 수식 7. surrogate loss를 clipping해서 actor loss 만들기
#             clipped_ratio = torch.clamp(ratio,
#                                         1.0 - hp.clip_param,
#                                         1.0 + hp.clip_param)
#             clipped_loss = clipped_ratio * advants_samples
#             actor_loss = -torch.min(loss, clipped_loss).mean()

#             loss = actor_loss + 0.5 * critic_loss # 0.5 - baseline 코드

#             critic_optim.zero_grad()
#             loss.backward(retain_graph=True) 
#             critic_optim.step()

#             actor_optim.zero_grad()
#             loss.backward()
#             actor_optim.step()


# def get_gae(rewards, masks, values):
#     rewards = torch.Tensor(rewards)
#     masks = torch.Tensor(masks)
#     returns = torch.zeros_like(rewards)
#     advants = torch.zeros_like(rewards)
    
#     running_returns = 0
#     previous_value = 0
#     running_advants = 0

#     # gamma = 0.99, lambda = 0.98
#     for t in reversed(range(0, len(rewards))):
#         running_returns = rewards[t] + (hp.gamma * running_returns * masks[t])
#         returns[t] = running_returns

#         # 논문에서 수식 10
#         running_delta = rewards[t] + (hp.gamma * previous_value * masks[t]) - \
#                                         values.data[t]
#         previous_value = values.data[t]
        
#         # 논문에서 수식 14 + lambda 추가
#         running_advants = running_delta + (hp.gamma * hp.lamda * \
#                                             running_advants * masks[t])
#         advants[t] = running_advants

#     advants = (advants - advants.mean()) / advants.std()
#     return returns, advants


# def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
#     mu, std = actor(torch.Tensor(states))
#     new_policy = log_prob_density(actions, mu, std)
#     old_policy = old_policy[batch_index]

#     # r_t (theta) = \pi_\theta (a_t | s_t) / \pi_{\theta_{old}} (a_t | s_t)
#     ratio = torch.exp(new_policy - old_policy)
#     surrogate_loss = ratio * advants
#     return surrogate_loss, ratio