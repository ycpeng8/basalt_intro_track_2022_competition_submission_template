import pickle
import gym
import minerl
import logging

MODEL = 'data/VPT-models/2x.model'
WEIGHTS = 'data/VPT-models/rl-from-early-game-2x.weights'

def main():
    logging.basicConfig(level=logging.DEBUG)

    env = gym.make('MineRLObtainDiamondShovel-v0')

    # Load the model
    agent_parameters = pickle.load(open(MODEL, "rb"))
    policy = agent_parameters["model"]["args"]["net"]["args"]
    pi_head = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head["temperature"] = float(pi_head["temperature"])
    

    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        action['ESC'] = 0
        obs, reward, done, _ = env.step(action)
        env.render()
