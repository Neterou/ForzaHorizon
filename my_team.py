# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        CaptureAgent.__init__(self, index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def get_visible_enemies(self, game_state):
        return [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

    def get_loaded_invaders(self, game_state, min_carry=3):
        enemies = self.get_visible_enemies(game_state)
        return [enemy for enemy in enemies
                if enemy.is_pacman and enemy.get_position() is not None
                and enemy.num_carrying >= min_carry]

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()
        features['successor_score'] = -len(food_list)

        # Closest food distance
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)
        else:
            features['distance_to_food'] = 0

        # Closest ghost distance (active ghosts only)
        enemies = self.get_visible_enemies(successor)
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        if ghosts:
            ghost_dist = min(self.get_maze_distance(my_pos, g.get_position()) for g in ghosts)
            features['ghost_near'] = 1 if ghost_dist <= 2 else 0
            features['ghost_distance'] = ghost_dist
        else:
            features['ghost_near'] = 0
            features['ghost_distance'] = 10

        # Penalize stopping
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -5,
            'ghost_near': -100,
            'ghost_distance': 1,
            'stop': -50,
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """A reflex agent focused on defending our territory."""

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        is_scared = my_state.scared_timer > 0

        # Defense indicator
        features['on_defense'] = 0 if my_state.is_pacman else 1

        enemies = self.get_visible_enemies(successor)
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]
        features['num_invaders'] = len(invaders)
        features['scared_escape_distance'] = 0
        features['scared_contact_risk'] = 0
        if invaders:
            dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            min_dist = min(dists)
            if is_scared:
                features['invader_distance'] = 0
                features['invader_chase_distance'] = 0
                features['invader_contact'] = 0
                features['scared_escape_distance'] = min_dist
                features['scared_contact_risk'] = 1 if min_dist <= 1 else 0
            else:
                features['invader_distance'] = min_dist
                features['invader_chase_distance'] = min_dist
                features['invader_contact'] = 1 if min_dist <= 1 else 0
        else:
            features['invader_distance'] = 0
            features['invader_contact'] = 0
            features['invader_chase_distance'] = 0
            features['scared_escape_distance'] = 0
            features['scared_contact_risk'] = 0

        loaded_invaders = self.get_loaded_invaders(successor)
        if loaded_invaders and not is_scared:
            loaded_dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in loaded_invaders]
            features['loaded_invaders'] = len(loaded_invaders)
            features['loaded_invader_distance'] = min(loaded_dists)
        else:
            features['loaded_invaders'] = 0
            features['loaded_invader_distance'] = 0

        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        configuration = game_state.get_agent_state(self.index).configuration
        prev_direction = configuration.direction if configuration else Directions.STOP
        reverse = Directions.REVERSE.get(prev_direction, Directions.STOP)
        features['reverse'] = 1 if action == reverse else 0

        return features

    def get_weights(self, game_state, action):
        del game_state, action
        return {
            'num_invaders': -1200,
            'invader_distance': -12,
            'invader_chase_distance': -20,
            'invader_contact': 300,
            'loaded_invaders': -100,
            'loaded_invader_distance': -25,
            'on_defense': 120,
            'stop': -150,
            'reverse': -4,
            'scared_escape_distance': 25,
            'scared_contact_risk': -500,
        }
