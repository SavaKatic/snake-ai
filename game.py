import pygame
import random
from enum import Enum
from collections import namedtuple
from abc import ABC, abstractmethod

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)

UNIT_SIZE = 20 # pixel size of each game block
SPEED = 2000

DIMENSIONS = {
    'width': 640,
    'height': 480
}

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

EVENTS = {
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN
}

class Game(ABC):
    """
    An abstract pygame class used to encapsulate methods that should be implemented in subclasses.
    """
    @abstractmethod
    def get_player(self):
        pass

    @abstractmethod
    def get_target_object(self):
        pass

    @abstractmethod
    def is_point_acquired(self):
        pass

    @abstractmethod
    def is_collision(self):
        pass

class SnakeGame(Game):
    """
    A class used to implement snake game rules.

    Attributes
    ----------
    screen: UI
        pygame UI screen
    snake : list
        list of points that represent a snake
    head : Point
        tuple of coordinates of snake head
    score: int
        current game score
    direction: Direction
        direction of the snake
    food: Point
        food placement
    iteration: int
        current frame iteration index, starts at 0
    """

    def __init__(self, screen):
        self._screen = screen
        self.restart()
        
    def restart(self):
        '''
        Restarts the game back to initial state.
        '''
        center = Point(self._screen.width / 2, self._screen.height / 2)
        self.direction = Direction.RIGHT
        
        self.score = 0
        self._head = center
        self._snake = [
            self._head, 
            Point(self._head.x - UNIT_SIZE, self._head.y),
            Point(self._head.x - (2 * UNIT_SIZE), self._head.y)
        ]
        
        self._place_target()
        self._iteration = 0


    def get_player(self):
        '''
        Gets points position of current snake.

        Returns:
            snake (list): position of snake
        '''
        return self._snake

    def get_target_object(self):
        '''
        Gets point position of current food spot.

        Returns:
            food (Point): position of food
        '''
        return self._food
        
    def get_snake_head(self):
        '''
        Gets point position of current snake head.

        Returns:
            head (Point): position of snake head
        '''
        return self._snake[0]
        
    def is_point_acquired(self):
        '''
        Checks if snake ate food.
        '''
        return self._head == self._food


    def play_step(self, action):
        '''
        Plays one game step of moving snake in certain direction and updating
        its size, food placement, score and UI.

        Parameters:
            action (list): an action to take
        '''

        self._iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._step(action)
        self._snake.insert(0, self._head)
        
        if self.is_collision() or self._is_stuck():
            return True, self.score, -10

        reward = 0
        if self.is_point_acquired():
            self.score += 1
            reward = 10
            self._place_target()
        else:
            self._snake.pop()
        
        self._screen.update()

        return False, self.score, reward


    def _place_target(self):
        '''
        Places food randomly in environment.
        If food placement happens to be on the snake, it repeats the process.
        '''
        self._food = Point(random.randint(0, (self._screen.width - UNIT_SIZE ) // UNIT_SIZE ) * UNIT_SIZE , random.randint(0, (self._screen.height - UNIT_SIZE ) // UNIT_SIZE ) * UNIT_SIZE)
        if self._food in self._snake:
            self._place_target()
        
    
    def _is_stuck(self):
        '''
        Checks if player (snake) is stuck in a loop
        '''
        return self._iteration > 100 * len(self._snake)


    def is_collision(self, point=None):
        '''
        Checks if snake collided with itself or boundary of screen.

        Parameters:
            point (Point): a point to check for collision
        '''
        if not point:
            point = self._head

        # check if snake hit itself
        if point in self._snake[1:]:
            return True

        # check if snake hit boundary
        if point.x > self._screen.width - UNIT_SIZE or point.x < 0 or point.y > self._screen.height - UNIT_SIZE or point.y < 0:
            return True
        
        return False

    def _step(self, action):
        '''
        Moves snake head in certain direction.

        Parameters:
            action (list): determines direction that should be taken
        '''

        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT ,Direction.UP]
        current_direction_index = directions.index(self.direction)

        if action == [1, 0, 0]: # stay same
            new_direction = directions[current_direction_index]
        elif action == [0, 1, 0]: # rotate right
            current_direction_index = (current_direction_index + 1) % 4
            new_direction = directions[current_direction_index]
        elif action == [0, 0, 1]: # rotate left
            current_direction_index = (current_direction_index - 1) % 4
            new_direction = directions[current_direction_index]

        self.direction = new_direction

        x = self._head.x
        y = self._head.y
        if self.direction == Direction.RIGHT:
            x += UNIT_SIZE
        elif self.direction == Direction.LEFT:
            x -= UNIT_SIZE
        elif self.direction == Direction.DOWN:
            y += UNIT_SIZE
        elif self.direction == Direction.UP:
            y -= UNIT_SIZE
            
        self._head = Point(x, y)


class UI:
    """
    A class used to represent pygame UI.

    Attributes
    ----------
    width : int
        width of the screen
    height : int
        height of the screen
    """
    def __init__(self):
        self.width = DIMENSIONS['width']
        self.height = DIMENSIONS['height']

        self.display = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

    def set_game(self, game):
        '''
        Sets game that will be played on screen.
        '''
        self.game = game

    def update(self):
        '''
        Updates screen based on new game object and target position.
        '''
        self.display.fill(BLACK)

        for point in self.game.get_player():
            pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x, point.y, UNIT_SIZE, UNIT_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.game.get_target_object().x, self.game.get_target_object().y, UNIT_SIZE, UNIT_SIZE))
        
        pygame.display.flip()

        self.clock.tick(SPEED)
