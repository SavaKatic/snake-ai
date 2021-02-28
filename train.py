import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from game import SnakeGame, UI, Direction, Point
from models import DQN
from agent import Agent

LR = 0.001 # learning rate
GAMMA = 0.9 # discount rate


class QTrain:
    """
    A class used to implement training functions.

    Attributes
    ----------
    lr: float
        learning rate of model
    gamma: float
        discount rate
    optimizer: optim.Adam
        implements Adam algorithm and L2 penalty
    criterion: nn.MSELoss
        implements MSE loss function
    """

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # for plotting
        self.fig = None
        self.ax = None

    def train_on_data(self, state, action, reward, next_state, game_over):
        '''
        Trains model on values of interest.

        Parameters:
            state (list): current game state
            action (list): action that determines where snake turns
            reward (int): reward based on action
            next_state (list): next game state
            game_over (bool): is game over?
        '''

        # create tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        predicted_action = self.model(state)

        updated_action = self.update_q_values(predicted_action, state, action, reward, next_state, game_over)
    
        self.optimizer.zero_grad()
        loss = self.criterion(updated_action, predicted_action)
        loss.backward()

        self.optimizer.step()

    def update_q_values(self, predicted_action, state, action, reward, next_state, game_over):
        '''
        Updates q values based on bellman equation.

        Parameters:
            predicted_action (list): action values predicted by model
            next_state (tensor): next state of the game
        '''
        target = predicted_action.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                maximum_expected_future_reward = torch.max(self.model(next_state[idx]))
                Q_new = reward[idx] + self.gamma * maximum_expected_future_reward

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        return target

    def save_checkpoint(self, file_name='model.pth'):
        '''
        Saves model weights as a checkpoint pth file locally.

        Parameters:
            file_name (str): name of checkpoint file
        '''
        model_folder_path = './checkpoints'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name)

    def load_checkpoint(self, file_name='model.pth'):
        '''
        Loads model weights that are saved as checkpoint .pth file.

        Parameters:
            file_name (str): name of checkpoint file
        '''

        model_folder_path = './checkpoints'
        if not os.path.exists(model_folder_path):
            return self.model

        file_name = os.path.join(model_folder_path, file_name)
        if not os.path.exists(file_name):
            return self.model

        model.load_state_dict(torch.load(file_name))
        return model


if __name__ == '__main__':
    scores = []
    mean_scores = []
    total_score = 0
    maximum_score = 0

    screen = UI()
    game = SnakeGame(screen)
    screen.set_game(game)

    model = DQN(11, 256, 3)
    train = QTrain(model, lr=LR, gamma=GAMMA)
    model = train.load_checkpoint()
    agent = Agent(model, train, game)

    # training loop
    while True:
        current_state = agent.get_state()
        next_action = agent.get_action(current_state)

        game_over, score, reward = game.play_step(next_action)
        new_state = agent.get_state()
        agent.train_over_sample(current_state, next_action, reward, new_state, game_over)

        agent.save_values(current_state, next_action, reward, new_state, game_over)

        if game_over:
            game.restart()
            agent.num_of_games += 1
            agent.train_over_batch()

            if score > maximum_score:
                # save checkpoint

                maximum_score = score
                agent.train.save_checkpoint()

            print('Game', agent.num_of_games, 'Score', score, 'Record:', maximum_score)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_of_games
            mean_scores.append(mean_score)
