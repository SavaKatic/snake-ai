import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, UI, Direction, Point

BATCH_SIZE = 1000
MEMORY_LIMIT = 100000

class Agent:
    """
    A class used to represent AI agent.

    Attributes
    ----------
    epsilon : float
        randomness (exploration vs exploitation)
    gamma : float
        discount rate
    num_of_games : list
        number of games played
    values : deque
        memory to save the data in used to train
    """

    def __init__(self, model, train, game):
        self.num_of_games = 0
        self.epsilon = 0
        self.values = deque(maxlen=MEMORY_LIMIT)
        self.model = model
        self.train = train
        self.game = game

    def get_state(self):
        '''
        Gets current state of the game.
        State depends on snake position relative to walls, itself and food.
        '''
        return np.array([
            self.is_danger_forward(),
            self.is_danger_right(),
            self.is_danger_left(),

            *self.get_game_directions(),
            
            self.game.get_target_object().x < self.game.get_snake_head().x,
            self.game.get_target_object().x > self.game.get_snake_head().x,
            self.game.get_target_object().y < self.game.get_snake_head().y,
            self.game.get_target_object().y > self.game.get_snake_head().y
        ], dtype=int)


    def is_danger_left(self):
        '''
        Checks if danger of collision is on the left turn.
        '''
        is_dir_left, is_dir_right, is_dir_up, is_dir_down = self.get_game_directions()
        point_left, point_right, point_up, point_down = self.get_points_around()

        return (is_dir_down and self.game.is_collision(point_right)) or \
            (is_dir_up and self.game.is_collision(point_left)) or \
            (is_dir_right and self.game.is_collision(point_up)) or \
            (is_dir_left and self.game.is_collision(point_down))


    def is_danger_right(self):
        '''
        Checks if danger of collision is on the right turn.
        '''
        is_dir_left, is_dir_right, is_dir_up, is_dir_down = self.get_game_directions()
        point_left, point_right, point_up, point_down = self.get_points_around()

        return (is_dir_up and self.game.is_collision(point_right)) or \
            (is_dir_down and self.game.is_collision(point_left)) or \
            (is_dir_left and self.game.is_collision(point_up)) or \
            (is_dir_right and self.game.is_collision(point_down))


    def is_danger_forward(self):
        '''
        Checks if danger of collision is forward.
        '''
        is_dir_left, is_dir_right, is_dir_up, is_dir_down = self.get_game_directions()
        point_left, point_right, point_up, point_down = self.get_points_around()

        return (is_dir_right and self.game.is_collision(point_right)) or \
            (is_dir_left and self.game.is_collision(point_left)) or \
            (is_dir_up and self.game.is_collision(point_up)) or \
            (is_dir_down and self.game.is_collision(point_down))


    def get_game_directions(self):
        '''
        Checks current direction of the game object (player).
        '''
        return self.game.direction == Direction.LEFT, \
            self.game.direction == Direction.RIGHT, \
            self.game.direction == Direction.UP, \
            self.game.direction == Direction.DOWN
    
    def get_points_around(self):
        '''
        Gets points on screen that are one step around all directions relative to player.
        '''
        head = self.game.get_snake_head()

        return Point(head.x - 20, head.y), \
               Point(head.x + 20, head.y), \
               Point(head.x, head.y - 20), \
               Point(head.x, head.y + 20)

    def save_values(self, state, action, reward, next_state, game_over):
        '''
        Saves values inside a deque.

        Parameters:
            state (list): current game state
            action (list): action that determines where snake turns
            reward (int): reward based on action
            next_state (list): next game state
            game_over (bool): is game over?
        '''
        self.values.append((state, action, reward, next_state, game_over))

    def train_over_batch(self):
        '''
        Trains model over a batch of data.
        '''
        batch = random.sample(self.values, BATCH_SIZE) if len(self.values) > BATCH_SIZE else self.values
        self.train.train_on_data(*zip(*batch))


    def train_over_sample(self, state, action, reward, next_state, game_over):
        '''
        Trains model over a single data sample.

        Parameters:
            state (list): current game state
            action (list): action that determines where snake turns
            reward (int): reward based on action
            next_state (list): next game state
            game_over (bool): is game over?
        '''
        self.train.train_on_data(state, action, reward, next_state, game_over)

    def get_action(self, state):
        '''
        Decides on exploration vs exploitation.
        Chooses next action based on it.
        
        Parameters:
            state (list): current game state
        '''

        # as num of games rises, epsilon should be lower
        self.epsilon = 80 - self.num_of_games
        next_action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # explore

            turn = random.randint(0, 2)
            next_action[turn] = 1
            return next_action

        # exploit

        state = torch.tensor(state, dtype=torch.float)
        next_action_prediction = self.model(state)
        turn = torch.argmax(next_action_prediction).item()
        next_action[turn] = 1

        return next_action
