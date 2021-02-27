import pygame
import random
from enum import Enum
from collections import namedtuple

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

UNIT_SIZE = 20
SPEED = 20

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

class Game:
    """
    A class used to implement game window and rules.

    Attributes
    ----------
    width : int
        width of the screen
    height : int
        height of the screen
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
    """

    def __init__(self):
        self.width = DIMENSIONS['width']
        self.height = DIMENSIONS['height']

        self.display = pygame.display.set_mode((self.width, self.height))

        center = Point(self.width / 2, self.height / 2)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.direction = Direction.RIGHT
        
        self.score = 0
        self.head = center
        self.snake = [
            self.head, 
            Point(self.head.x - UNIT_SIZE, self.head.y),
            Point(self.head.x - (2 * UNIT_SIZE), self.head.y)
        ]
        
        self._place_food()
        

    def play_step(self):
        '''
        Plays one game step of moving snake in certain direction and updating
        its size, food placement, score and UI.
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            # if key was pressed
            if event.type == pygame.KEYDOWN and event.key in EVENTS:
                self.direction = EVENTS[event.key]
        

        self._step(self.direction)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_screen()
        self.clock.tick(SPEED)

        return game_over, self.score


    def _place_food(self):
        '''
        Places food randomly in environment.
        If food placement happens to be on the snake, it repeats the process.
        '''
        self.food = Point(random.randint(0, (self.width - UNIT_SIZE ) // UNIT_SIZE ) * UNIT_SIZE , random.randint(0, (self.height - UNIT_SIZE ) // UNIT_SIZE ) * UNIT_SIZE)
        if self.food in self.snake:
            self._place_food()
    

    def _is_collision(self):
        '''
        Checks if snake collided with itself or boundary of screen.
        '''
        # check if snake hit itself
        if self.head in self.snake[1:]:
            return True

        # check if snake hit boundary
        if self.head.x > self.width - UNIT_SIZE or self.head.x < 0 or self.head.y > self.height - UNIT_SIZE or self.head.y < 0:
            return True
        
        return False

    def _step(self, direction):
        '''
        Moves snake head in certain direction.
        '''
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += UNIT_SIZE
        elif direction == Direction.LEFT:
            x -= UNIT_SIZE
        elif direction == Direction.DOWN:
            y += UNIT_SIZE
        elif direction == Direction.UP:
            y -= UNIT_SIZE
            
        self.head = Point(x, y)

    
    def _update_screen(self):
        '''
        Updates screen based on new snake and food position.
        '''
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, UNIT_SIZE, UNIT_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, UNIT_SIZE, UNIT_SIZE))
        
        pygame.display.flip()
            

if __name__ == '__main__':
    pygame.init()

    game = Game()
    
    end = False
    while not end:
        end, score = game.play_step()
        
    print('Game Score', score)
        
        
    pygame.quit()
