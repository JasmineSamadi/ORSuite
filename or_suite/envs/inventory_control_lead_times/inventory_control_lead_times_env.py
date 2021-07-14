import gym
import numpy as np
import sys

from .. import env_configs


class InventoryControlLeadTimesEnvironment(gym.Env):

    # note I just chose some default values
    def __init__(self, config=env_configs.inventory_control_lead_times_default_config, **kwargs):
        # parameter of exponenetial demand, 1/alpha is mean demand
        self.alpha = config['alpha']
        self.h = config['h']  # holding cost
        self.p = config['p']  # lost sales penalty
        self.L = config['L']  # lead time
        self.length = config['length']  # how many periods to run simulation

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # optimal constant order amount
        self.opt_const = (1/self.alpha)*(1-(self.h/(2*self.p+self.h))**(1/2))
        # optimal constant order policy value
        self.opt_const_val = (1/self.alpha) * \
            ((self.h*(2*self.p + self.h))**(1/2)-self.h)
        self.steps = 0  # keeps track of how many steps have been done in this episode
        # first entry is current inventory, remaining entries are outstanding orders, initialize all to optimal constant order amount
        self.state = np.zeros(self.L + 1) + self.opt_const
        self.action_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(self.L + 1,))
        self.seed()
        metadata = {'render.modes': ['ansi']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):  # note action must be of an np array
        assert self.action_space.contains(action)

        # update on hand inventory
        on_hand = self.state[0] + self.state[1]

        # simulate demand
        demand = self.np_random.exponential(self.alpha)

        # post demand inventory
        on_hand_post = max(on_hand - demand, 0)

        # incur costs
        lost_sales = -min(on_hand - demand, 0)
        reward = -(self.h * on_hand_post + self.p * lost_sales) / self.length

        # update pipeline vector
        next_state = np.zeros(self.L + 1)
        next_state[1:-1], next_state[-1] = self.state[2:], action
        next_state[0] = on_hand_post
        self.state = next_state
        self.steps = self.steps + 1

        done = (self.steps >= self.length)
        return self.state, reward, done, {}

    def render(self, mode='ansi'):
        outfile = sys.stdout if mode == 'ansi' else super(
            InventoryControlLeadTimesEnvironment, self).render(mode=mode)
        outfile.write(np.array2string(self.state))

    def reset(self):
        self.steps = 0  # keeps track of how many steps have been done in this episode
        # first entry is current inventory, remaining entries are outstanding orders, initialize all to optimal constant order amount
        self.state = np.zeros(self.L + 1) + self.opt_const
        return self.state
