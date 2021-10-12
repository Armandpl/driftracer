import gym
import drift_gym

env = gym.make("Real-v0")

env.reset()
print("let's go")

for i in range(100):
    action = \
        env.action_space.sample()
    env.step(action)

env.close()
