import gym
import drift_gym

if __name__ == "__main__":
    env = gym.make("Real-v0")

    env.reset()
    print("let's go")

    for i in range(1000):
        try:
            action = 1.0
            env.step(action)
        except KeyboardInterrupt:
            env.close()
            input("press any key to continue")
            env.reset()

    env.close()
