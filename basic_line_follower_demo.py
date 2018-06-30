'''
Demonstration of line follower modules
'''

import pygame
import numpy as np
import os
import sys

from follower import *
from closed_path import *

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (25, 25)
pygame.init()
# if sound used
# pygame.mixer.init()

### assets
#sim_dir = os.path.dirname(__file__)

BLACK = (0,   0,   0  )
WHITE = (255, 255, 255)

### create track and screen
size = (700, 700)
screen_center = [size[0]/2, size[1]/2]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("closed track")

path = ClosedPath(npts = 500, bias = 200, order = 4, line_width = 3, center = [size[0]/2, size[1]/2], amp = 140)
screen.fill((255,255,255))
path.draw_path(screen, BLACK)
pygame.display.flip()

### initialize foll
initt = np.random.rand() * 2 * np.pi
initx = path.get_x(initt)
inity = path.get_y(initt)
init_phi = path.tangent_angle(initt)
foll = Follower(initx, inity, init_phi, width = 30, length = 30, h_offset = 10, sens_w_offset = 0.3)
foll.reset_position(initx, inity, init_phi)

clock = pygame.time.Clock()

max_epis = 50

for epi in range(max_epis):
    steps = 0
    path.draw_path(screen, BLACK)
    pygame.display.flip()
    running = True

    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Flag that we are done so we exit this loop

        # --- Game logic should go here ---> update
        #foll.rotate_left(1)
        #all_sprites.update()
        #foll.dual_wheel_move(2*np.random.rand(), 2*np.random.rand())
        #foll.dual_wheel_move(2, 2)
        foll.manual_rotate()
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_z]:
            initt = np.random.rand() * 2 * np.pi
            initx = path.get_x(initt)
            inity = path.get_y(initt)
            init_phi = path.tangent_angle(initt)
            foll.reset_position(initx, inity, init_phi)

        foll.update()

        # --- Drawing code should go here ---> draw
        screen.fill((255,255,255))
        path.draw_path(screen, BLACK)
        foll.follower_draw(screen)
        #print(foll.observation(grad, [foll.h_point, foll.t_point])[0])

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        steps += 1

    foll.reset_position(300, 150, np.random.rand() * 2 * np.pi)

#Once we have exited the main program loop we can stop the game engine:
pygame.quit()
