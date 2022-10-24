from email import policy
import logging
import coloredlogs
import pickle

import aicrowd_gym
import minerl

from config import EVAL_EPISODES, EVAL_MAX_STEPS
from openai_vpt.agent import MineRLAgent

coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = 'MineRLObtainDiamondShovel-v0'
MODEL = 'train/VPT-models/2x.model'
WEIGHTS = 'train/VPT-models/rl-from-early-game-2x.weights'

def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    model = pickle.load(open(MODEL, "rb"))
    policy = model["model"]["args"]["net"]["args"]
    pi_head = model["model"]["args"]["pi_head_opts"]
    pi_head["temperature"] = float(pi_head["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy, pi_head_kwargs=pi_head)
    agent.load_weights(WEIGHTS)

    for i in range(EVAL_EPISODES):
        obs = env.reset()
        agent.reset()
        done = False
        
        for step_counter in range(EVAL_MAX_STEPS):

            # Step your model here.
            # Currently, it's doing random actions
            minerl_action = agent.get_action(obs)

            obs, reward, done, info = env.step(minerl_action)
            # env.render()

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    main()
