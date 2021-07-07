import drift_gym
import gym
import wandb

from td3_torch import Agent

if __name__ == '__main__':

    config = dict(
        agent="td3",
        alpha=0.001,
        beta=0.001,
        tau=0.005,
        batch_size=100,
        layer1_size=400,
        layer2_size=300,
        n_games=1000,
        throttle=0.03,
        dt=0.05,
        horizon=200
    )

    with wandb.init(project="driftracer",
                    config=config,
                    job_type="train") as run:

        config = run.config

        env = gym.make('Real-v0',
                       throttle=config.throttle,
                       dt=config.dt,
                       horizon=config.horizon)

        agent = Agent(alpha=config.alpha, beta=config.beta,
                      input_dims=env.observation_space.shape, tau=config.tau,
                      env=env, batch_size=config.batch_size,
                      layer1_size=config.layer1_size,
                      layer2_size=config.layer2_size,
                      n_actions=env.action_space.shape[0])

        print("drift drift")
        for i in range(config.n_games):
            observation = env.reset()
            done = False
            score = 0
            try:
                while not done:
                    action = agent.choose_action(observation)[0]
                    observation_, reward, done, info = env.step(action)

                    agent.remember(observation,
                                   action,
                                   reward,
                                   observation_,
                                   done)
                    agent.learn()

                    score += reward
                    observation = observation_
                    wandb.log({"action": action,
                               "episode": i,
                               "reward": reward})

            except KeyboardInterrupt:
                env.close()
                input("press key to continue")

            print('episode ', i, 'score %.1f' % score)
