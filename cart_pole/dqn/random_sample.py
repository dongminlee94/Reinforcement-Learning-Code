import gym

env = gym.make('CartPole-v1')

state = env.reset()

for _ in range(1000):
    env.render()  # Show the initial board
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)