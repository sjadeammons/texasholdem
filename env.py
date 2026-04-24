# %%
from __future__ import annotations
from gym_env.env import HoldemTable


"""Groupier functions"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from gymnasium.utils import EzPickle

from gym_env.cycle import PlayerCycle
from gym_env.enums import Action, Stage
from gym_env.rendering import PygletWindow, WHITE, RED, GREEN, BLUE
from tools.hand_evaluator import get_winner
from tools.helper import flatten

import argparse 
from copy import copy
from collections.abc import Callable
from typing import Any, Generic, Iterator, Union
from pettingzoo.utils.env import AECEnv
#from pettingzoo.utils import AgentSelector

from agents.agent_random import Player as RandomPlayer 
from pettingzoo.utils.env import (
    ActionType,
    AECEnv,
    AECIterable,
    AECIterator,
    AgentID,
    ObsType,
    AgentID
)
from marco_polo.envs.TexasHoldEm.params import TexasHoldEmParams

logging.basicConfig(level=20)
log = logging.getLogger(__name__)
        

def get_env_class(args: argparse.Namespace) -> Callable[..., Any]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env objects
    """

    return TexasHoldEmDecisionTree

class TexasHoldEmDecisionTree(HoldemTable, AECEnv):
    metadata = {'render_modes': ['human'], "name": "gym_to_aec_v0"}
        
    def __init__(
            self, 
            num_players = 2,
            initial_stacks= 10, 
            small_blind=1,
            big_blind=2,
            render=False,
            funds_plot=True,
            max_raises_per_player_round=2,
            use_cpp_montecarlo=False,
            raise_illegal_moves=False,
            calculate_equity=False,
            render_mode=None,
            yaml_rewards = None
            ):
        
        super().__init__(initial_stacks, small_blind, big_blind, render, funds_plot,
                max_raises_per_player_round, use_cpp_montecarlo, raise_illegal_moves,
                calculate_equity)
        
        self.num_players = num_players

        # Define agents
        for i in range(num_players):
            player = RandomPlayer(name = f"Player_{i}")
            del player.autoplay
            self.add_player(player)
        self.possible_agents = [a.name for a in self.agents]

        # Set spaces
        self.action_spaces = {agent: self.act_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self.obs_space for agent in self.possible_agents}

        # Not sure if this should be here...
        max_steps_after_raiser = (self.max_raises_per_player_round - 1) * len(self.agents) - 1
        self.player_cycle = PlayerCycle(self.agents, dealer_idx=-1, max_steps_after_raiser=max_steps_after_raiser,
                                        max_steps_after_big_blind=len(self.agents),
                                        max_raises_per_player_round=self.max_raises_per_player_round)
        
        self.agent_selection = self.agents[self.player_cycle.start_idx].name

        # print(f"YAML REWS {yaml_rewards}", flush = True)
        #self.augment(yaml_rewards)
        self.__dict__.update(yaml_rewards)
        
        #print(f"AGENT SELECTION: {(self.agent_selection)} in INIT", flush = True)
    
    def reset(self, seed = None, options = None):

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # Reset gym env
        super().reset()

        self.agent_selection = self.agents[self.player_cycle.start_idx].name
        
        
    def step(self, action):
        
        print(f"LEGAL MOVE PROBS INCLUDING ILLEGAL: {list(action)}", flush = True)

        ### FILTER ACTION FOR LEGAL MOVES ONLY

        possible_acts = [Action(a) for a in range(8)]

        # set probability of choosing illegal moves to 0
        legal_move_probs_only = []
        for i in range(len(action)):
            if possible_acts[i] in self.legal_moves:
                legal_move_probs_only += [action[i]]
            else:
                legal_move_probs_only += [0]

        print(f"LEGAL MOVE PROBS BEFORE NORMALIZING: {list(legal_move_probs_only)}", flush = True)

        # normalize the remaining probs, and sample from that to get the action
        legal_move_probs_only = np.array(legal_move_probs_only)/np.sum(legal_move_probs_only)
        
        action = np.random.choice(possible_acts, p = legal_move_probs_only)

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_last_agent_step(action)
            return

        curr_player = self.agents[self.player_cycle.idx]
        curr_player_idx = self.player_cycle.idx

        ### TAKE STEP IN GYM
        obs, reward, done, info = super().step(action)

        if done and (curr_player.name == "Player_0") and (Action(7) in self.legal_moves):
            print(f"LEGAL MOVE PROBS AFTER NORMALIZING: {list(legal_move_probs_only)}", flush = True)

        # update agent_selection to current player
        self.agent_selection = curr_player.name

        ### Update AEC structures
        # self.rewards[self.agent_selection] = reward # <--- we will update the rewards below.
        self.terminations[self.agent_selection] = done
        self.infos[self.agent_selection] = info
        self._cumulative_rewards[self.agent_selection] = 0
        
        ###################################################
        ###                  REWARDS                    ###
        ###################################################
        log.info(f"REWARDS BEFORE: {self.rewards}")
        its_someones_first_hand = sum(self.first_action_for_hand) > 1

        # if done:
        #     # Reward winner and all losers at end of tournament
        #     for i, player in enumerate(self.possible_agents):

        #         won = 1 if (i == self.winner_ix) else -1
        #         # give winner remaining chips
        #         diff = ((self.initial_stacks * len(self.agents)) -  self.funds_history.iloc[-2, self.winner_ix]) * won
        #         bonus = won * self._END_TOURNEY_BONUS
        #         tourney_rew = diff + bonus
        #         self.rewards[player] += tourney_rew #self.initial_stacks * len(self.agents) * won
        #         log.info(f"{player} REWARD FOR END OF TOURNAMENT: {tourney_rew} (chip diff {diff} + bonus of {bonus})")

        # # Reward all agents for differences in chips between start of this hand and last hand
        # elif (len(self.funds_history) > 1) and its_someones_first_hand:

        #     print(f"Last 2 rows of Funds History:", flush = True)
        #     print(self.funds_history.tail(2), flush = True)
        #     for i, player in enumerate(self.possible_agents):
 
        #         hand_rew = self.funds_history.iloc[-1, i] - self.funds_history.iloc[-2, i]
        #         self.rewards[player] += hand_rew
        #         #log.info(f"{player} REWARD FOR BEGINNING HAND: {hand_rew}")
        #         self.first_action_for_hand[i] = False
        # else:
        #     pass

        # ACTION REWARDS
        if self.agent_selection == "Player_0":
            
            # print(f"ACTION: {Action.action}", flush = True)
            # print(action == Action.ALL_IN, flush = True)
            act_reward = 0
            match Action(action):
                case Action.FOLD: # FOLD
                    act_reward = self._FOLD_REW

                case Action.CHECK:
                    act_reward = self._CHECK_REW

                case Action.CALL:
                    act_reward = self._CALL_REW

                case Action.RAISE_3BB:
                    act_reward = self._RAISE_3BB_REW

                case Action.RAISE_HALF_POT:
                    act_reward = self._RAISE_HALF_POT_REW

                case Action.RAISE_POT:
                    act_reward = self._RAISE_POT_REW

                case Action.RAISE_2POT:
                    act_reward = self._RAISE_2POT_REW

                case Action.ALL_IN:
                    act_reward = self._ALL_IN_REW

            self.rewards["Player_0"] += act_reward

            log.info(f"Player_{self.acting_agent} ACTION REWARD: {act_reward}") 

        # Report rewards per player
        rews_list = []
        for agent in self.possible_agents:
            rew = self.rewards[agent]
            sign = ""
            if rew >0:
                sign = "+"
            rews_list += [agent + ": " + sign + str(rew)]
        log.info(f"--------------> TOTAL REWARDS: {rews_list}")
        log.info(f"-------------------------------------------------------------------------------------------")
            
    def observe(self, agent):
        return self.obs_space.sample() # Return actual obs

    def render(self):
        return self.render()

    def agent_iter(self, max_iter: int = 2**63) -> AECIterable:
        """Yields the current agent (self.agent_selection).

        Needs to be used in a loop where you step() each iteration.
        """
        return OurAECIterator(self, max_iter)

    def _check_game_over(self):
    
        # Check if someone has won _HANDS_TO_WIN many hands
        someone_won_enough_hands = False
        if self._HANDS_TO_WIN in self.hands_won:
            someone_won_enough_hands = True

        # Check if only one player has money left
        player_alive = []
        for idx, player in enumerate(self.agents):
            if (player.stack > 0) or (self.player_cycle.out_of_cash_but_contributed[idx]):
                player_alive.append(True)
        remaining_players = sum(player_alive)

        if (remaining_players < 2) or someone_won_enough_hands:
            self._game_over()
            return True

        return False


# Created self.hands_won, and update it after call to super step. Overwrote _check_game_over to check if an entry in hands_won exceeds
# the new parameter _HANDS_TO_WIN. If so, it calls game_over function and returns true.
    

class OurAECIterator(Iterator[AgentID], Generic[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], max_iter: int):
        #print("INIT IN ITERATOR", flush = True)
        self.env = env
        self.iters_til_term = max_iter

    def __next__(self) -> AgentID:
        #print("NEXT IN ITERATOR", flush = True)
        if self.env._check_game_over() or self.iters_til_term <= 0:
            raise StopIteration
        self.iters_til_term -= 1
        #print(f"AGENT SELECTION IN ITERATOR: {self.env.agent_selection}", flush = True)
        return self.env.agent_selection

    def __iter__(self) -> AECIterator[AgentID, ObsType, ActionType]:
        return self


# %%

            
            


