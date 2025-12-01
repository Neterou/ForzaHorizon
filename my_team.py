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
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

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
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
    def get_loaded_invaders(self, game_state, min_carry=3):
        enemies = self.get_visible_enemies(game_state)
        return [enemy for enemy in enemies
                if enemy.is_pacman and enemy.get_position() is not None
                and enemy.num_carrying >= min_carry]
    
    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0, 'stop': -100.0, 'reverse': -10.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = super().get_features(game_state, action)
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True, but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Avoid visible ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

        ghost_dists = []

        for g in ghosts:
            ghost_dists.append(self.get_maze_distance(my_pos, g.get_position()))

        # Default values
        features['ghost_distance'] = 10

        if len(ghost_dists) > 0:
            min_dist = min(ghost_dists)

            features['ghost_distance'] = min_dist

        # Encourage returning home when carrying multiple pellets
        carried = successor.get_agent_state(self.index).num_carrying
        features['carrying'] = carried

        # Home urgency feature
        home_pos = self.start
        dist_home = self.get_maze_distance(my_pos, home_pos)
        if carried > 2:
            features['home_urgency'] = carried * (dist_home/20)

    
        capsules_now = self.get_capsules(game_state)
        features['capsule_distance'] = 999
        features['will_get_capsule'] = 0

        if len(capsules_now) > 0:
            cap_dist = min(self.get_maze_distance(my_pos, c) for c in capsules_now)
            features['capsule_distance'] = cap_dist

            successor_pos = successor.get_agent_state(self.index).get_position()

            # Provide strong incentive if the action grabs a capsule
            if successor_pos in capsules_now:
                features['will_get_capsule'] = 10
        
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -2, 'ghost_distance': 5, 'carrying': 10, 'home_urgency': -6, 'capsule_distance': -1,'stop': -100, 'eat_capsule': 300, 'capsule_attraction': 80, 'reverse': -2}
class DefensiveReflexAgent(ReflexCaptureAgent):
    """Defensive agent tuned to chase invaders, handle scared mode, and guard objectives."""

    def get_features(self, game_state, action):
        # Base features: successor_score, stop, reverse
        features = super().get_features(game_state, action)
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        is_scared = my_state.scared_timer > 0

        features['on_defense'] = 0 if my_state.is_pacman else 1

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        features['num_invaders'] = len(invaders)

        # Initialize chase-related features
        features['invader_distance'] = 0
        features['invader_contact'] = 0
        features['invader_chase_distance'] = 0
        features['scared_escape_distance'] = 0
        features['scared_contact_risk'] = 0

        if invaders:
            dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            min_dist = min(dists)

            if is_scared:
                # When scared we focus on escaping
                features['scared_escape_distance'] = min_dist
                features['scared_contact_risk'] = 1 if min_dist <= 1 else 0
            else:
                # Otherwise we chase invaders directly
                features['invader_distance'] = min_dist
                features['invader_chase_distance'] = min_dist
                features['invader_contact'] = 1 if min_dist <= 1 else 0

        loaded_invaders = [e for e in invaders if e.num_carrying >= 3]

        if loaded_invaders and not is_scared:
            ldists = [self.get_maze_distance(my_pos, e.get_position()) for e in loaded_invaders]
            features['loaded_invaders'] = len(loaded_invaders)
            features['loaded_invader_distance'] = min(ldists)
        else:
            features['loaded_invaders'] = 0
            features['loaded_invader_distance'] = 0

        capsules = self.get_capsules_you_are_defending(successor)
        if capsules:
            features['capsule_distance'] = min(self.get_maze_distance(my_pos, c) for c in capsules)
        else:
            features['capsule_distance'] = 0

        midline = successor.get_walls().width // 2
        features['center_defense'] = abs(my_pos[0] - midline)

        return features


    def get_weights(self, game_state, action):
        return {
            # Prioritize stopping invaders
            'num_invaders': -1200,
            'invader_distance': -20,
            'invader_chase_distance': -20,
            'invader_contact': 300,

            # Focus extra attention on loaded invaders
            'loaded_invaders': -150,
            'loaded_invader_distance': -30,

            # Behavior while scared
            'scared_escape_distance': 25,
            'scared_contact_risk': -500,

            # Structural defense priorities
            'on_defense': 150,
            'capsule_distance': -2,
            'center_defense': -1,

            # Movement penalties inherited from base agent
            'stop': -100.0,
            'reverse': -10.0,

            # Maintain score awareness
            'successor_score': 1.0,
        }

