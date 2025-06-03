import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class CartPoleEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.pole_com_length = 0.5 # half of pole length
        self.input_force = 10.0
        self.polemass_length = self.masspole * self.pole_com_length
        
        self.tau = 0.02  # seconds between state updates

        # Limits
        self.theta_threshold_radians =  math.pi # Half rotation
        self.x_threshold = 5.0

        high = np.array([ self.x_threshold * 2, np.finfo(np.float32).max,
                         self.theta_threshold_radians*2, np.finfo(np.float32).max], dtype=np.float32)
        
        # Spaces
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = left, 1 = right

        # State
        self.state = None
        self.steps_beyond_done = None

        # Rendering
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self.cart_y = 300

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.input_force if action == 1 else -self.input_force

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        """
        Equations of Motion for the CartPole system:
        
        θ̈ = [ g * sin(θ) + cos(θ) * ( -F - [m_p*l*θ̇²* sin(θ)] ) / (m_c + m_p) ] /
             [ l * (4/3 - (m_p * cos²(θ)) / (m_c + m_p)) ]
        
        ẍ = [ F + m_p * l * (θ̇² * sin(θ) - θ̈ * cos(θ)) ] / (m_c + m_p)
        
        Where:
        θ  = pole angle (from vertical)
        θ̇ = angular velocity
        θ̈ = angular acceleration
        ẍ = cart acceleration
        F  = force applied to cart
        g  = gravity
        l  = half the pole length (center of mass)
        m_c = mass of the cart
        m_p = mass of the pole
        """

        # F + m_p*l*(θ̇² * sin(θ)) -> Made this a temporary variable as this is repeated
        temporary = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass 
        # Theta acceleration
        theta_acc = (
            self.gravity*sintheta - costheta*(temporary) / 
            (self.pole_com_length*((4.0/3.0) - (self.masspole*costheta**2/self.total_mass)))
            )
        # X acceleration
        x_acc = temporary - (self.polemass_length*(theta_acc*costheta)/self.total_mass)

        new_x = x + x_dot* self.tau # x = x + ut
        new_x_dot = x_dot + x_acc * self.tau # v = u + at
        new_theta = theta + theta_dot * self.tau # θ = θ + ωt
        new_theta_dot = theta_dot + theta_acc * self.tau # ω = ω + αt

        self.state = (new_x, new_x_dot, new_theta, new_theta_dot)

        game_over_state = (
            x < -self.x_threshold or
            x > self.x_threshold or
            theta < -self.theta_threshold_radians or
            theta > self.theta_threshold_radians
        )

        if not game_over_state:
            reward = 1.0
        else:
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, bool(game_over_state), False, {}
    
    def render(self, mode="human", fps=30):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Custom CartPole")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Extract cart position
        cart_width = 50
        cart_height = 30

        cart_range_width = self.x_threshold * 2
        scale = self.screen_width / cart_range_width
        cartx = int(self.state[0] * scale + self.screen_width / 2) # 0 -> Center of screen, -5 -> Left, 5 -> Right
        true_cartx = cartx - cart_width // 2

        # Draw the cart
        pygame.draw.rect(self.screen, (0, 0, 0), (true_cartx, self.cart_y, cart_width, cart_height))

        # Draw pole
        pole_len = scale * (2 * self.pole_com_length)
        theta = self.state[2]
        pole_x = cartx
        pole_y = self.cart_y

        end_x = pole_x + pole_len * math.sin(theta)
        end_y = pole_y - pole_len * math.cos(theta)

        pygame.draw.line(self.screen, (255, 0, 0), (pole_x, pole_y), (end_x, end_y), 6)

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None