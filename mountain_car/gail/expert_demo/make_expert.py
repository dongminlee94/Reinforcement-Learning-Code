import gym
import numpy as np

demo_count = 5000

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / 20 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * 20
    return state_idx

def main():    
    env = gym.make('MountainCar-v0')

    q_table = np.load(file="expert_q_table.npy")
    print("q_table.shape", q_table.shape)

    demonstrations = []
    episodes, scores = [], []

    for episode in range(100000):
        state = env.reset()
        temp = []
        score = 0

        while True:
            # env.render()
            state_idx = idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)

            if done:
                break

            temp.append((state[0], state[1], action))

            score += reward
            state = next_state

        if score > -120:
            print('{} episode score is {:.2f}'.format(episode, score))
            if len(demonstrations) < demo_count:
                for i in range(len(temp)):
                    demonstrations.append((temp[i][0], temp[i][1], temp[i][2]))
            else: break
        
    demo = np.array(demonstrations, float)
    print("demo.shape", demo.shape)

    np.save("expert_demo", arr=demo)

if __name__ == '__main__':
    main()