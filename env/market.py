import gym
import numpy as np
import pandas as pd

from gym import spaces


class MarketEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    actions = {'SELL': 0, 'HOLD': 1}
    params = {
        'forex': 'EURUSD',
        'data_dir': '/Users/mreiter/Desktop/thesis/deep_rl/trading/block_lob/eurusd',
        'file_name': 'DAT_ASCII_EURUSD_M1_',
        'start': 2010,
        'end': 2015,
        'offset': 20,
        'inventory': 100000,
        'impact': 1e-12,  # 1/141450228.13,
        'resilience': 100,
        'tpm_low': 1,
        'tpm_high': 2
    }

    def __init__(self, params):
        self.start = params['start']
        self.end = params['end']

        self.data_dir = params['data_dir']
        self.file_name = params['file_name']

        self.quotes = pd.read_csv('{}/{}{}.csv'.format(self.data_dir,
                                                       self.file_name,
                                                       str(self.start)), header=None)

        self.offset = params['offset']
        self.index = self.offset
        self.date = self.quotes.iloc[self.index, 0]
        self.time = self.quotes.iloc[self.index, 1]

        self.current_step = 0
        self.episode_length = len(self.quotes.loc[self.quotes[0] == self.date])

        # market related parameters
        self.bid = self.quotes.iloc[self.index, 2]
        self.impact = params['impact']
        self.res = params['resilience']
        self.tpm_high = params['tpm_high']
        self.tpm_low = params['tpm_low']

        # inventory related parameters
        self.inventory = params['inventory']
        self.sold = 0
        self.slippage = 0

        # define the action and observation spaces
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 20), dtype=np.float16)

    def reset(self):
        # update the quotes if exhausted current file
        if self.index == len(self.quotes):
            current_year = int(str(self.date)[:4])

            # if we are done, set quotes to None
            if current_year == self.end:
                self.quotes = None
            else:
                year = current_year + 1
                self.index = self.offset

                self.quotes = pd.read_csv('{}/{}{}.csv'.format(self.data_dir,
                                                               self.file_name,
                                                               str(year)), header=None)

        self.date = self.quotes.iloc[self.index, 0]
        self.time = self.quotes.iloc[self.index, 1]

        self.current_step = 0
        self.episode_length = len(self.quotes.loc[self.quotes[0] == self.date])

        self.bid = self.quotes.iloc[self.index, 2]
        self.sold = 0
        self.slippage = 0

        return self._get_state()

    def _get_state(self):
        # for notational convenience, keep the last 20 minutes of OHLC data in a list
        OHLC = []

        for i in range(2, 6):
            OHLC.append(
                self.quotes.iloc[(self.index - self.offset):self.index, i]
            )

        # observation consists of the last 20 minutes of OHLC data
        obs_frame = np.array([
            OHLC[0].values / OHLC[0].max(),
            OHLC[1].values / OHLC[1].max(),
            OHLC[2].values / OHLC[2].max(),
            OHLC[3].values / OHLC[3].max(),
        ])

        # additional data and scale each value to between 0-1
        obs = np.append(
            obs_frame,
            [np.append([self.current_step / self.episode_length,
                        self.sold / self.inventory,
                        self.slippage / self.inventory,
                        self.bid / self.quotes.iloc[self.index, 3], ],
                       np.zeros((1, 16)))],
            axis=0)

        return obs

    def step(self, action):
        # if we are at the end of the training process
        if self.quotes is None:
            return None, None, True, 'training is done ... the training set is exhausted'

        # execute the trade, update the new bid price and record slippage and pnl (the reward)
        reward = self._execute_trade(action)

        # update the counters
        self.index += 1
        self.current_step += 1

        # get the state
        obs = self._get_state()

        # check if done
        if self.current_step == self.episode_length:
            done, info = True, 'the trading day is done'
        else:
            # update the date and time
            self.date = self.quotes.iloc[self.index, 0]
            self.time = self.quotes.iloc[self.index, 1]

            done, info = False, 'the trading day continues'

        return obs, reward, done, info

    def _execute_trade(self, action):
        direction = action[0]  # sell or hold
        amount = action[1] * (self.inventory - self.sold)  # % of remaining inventory

        # update the amount sold
        self.sold += amount

        # get the elapsed time between each step ... its usually 1 minute but sometimes it's more b/c of missing data
        dt = (self.quotes.iloc[self.index, 1] - self.quotes.iloc[self.index - 1, 1]) / 100

        # the opening price which the bid price will try to revert to
        open = self.quotes.iloc[self.index, 2]

        # initialize the bid price
        bid = open + (self.bid - open) * np.exp(-self.res * dt)

        if direction == 0:
            # pnl without transaction costs
            pure_pnl = self.bid * amount

            # update the slippage ... capped at 25% of pure_pnl
            slippage = min(self.impact * amount ** 2, 0.25 * pure_pnl)
            self.slippage += slippage

            # pnl is the price realized at the bid level minus the slippage incurred
            pnl = pure_pnl - slippage

            # uniform sampler to get the number of trades completed per minute ... ranging uniformly from 1/s to 2/s
            tpm = dt * np.random.randint(60 * self.tpm_low, 60 * self.tpm_high + 1)

            # update the bid price to be impacted by trading activity
            for i in range(1, int(tpm) + 1):
                bid -= self.impact * (amount / tpm) * np.exp(-self.res * dt * (1 - i / tpm))

        # update the bid price
        self.bid = bid

        return pnl

    def render(self, mode='human', close=False):
        pass
