import pygame
import random
import numpy as np
import collections
import tensorflow as tf
from keras import layers, models
import cv2

# --- Core Simulation Parameters ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
# Parameters for the CNN model
IMG_WIDTH = 84
IMG_HEIGHT = 84

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
DARK_GRAY = (50, 50, 50)
BLUE = (0, 0, 255)


# --- Simulation Classes (Modified for RL) ---

class TrafficLight:
    """Represents a traffic light at the intersection. State is controlled externally."""
    def __init__(self, x, y, orientation, initial_state='red'):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.state = initial_state

    def draw(self, surface):
        """Draws the traffic light on the screen."""
        if self.orientation == 'vertical':
            pygame.draw.rect(surface, DARK_GRAY, (self.x, self.y, 30, 90))
            pygame.draw.circle(surface, RED if self.state == 'red' else GRAY, (self.x + 15, self.y + 15), 10)
            pygame.draw.circle(surface, YELLOW if self.state == 'yellow' else GRAY, (self.x + 15, self.y + 45), 10)
            pygame.draw.circle(surface, GREEN if self.state == 'green' else GRAY, (self.x + 15, self.y + 75), 10)
        else:  # Horizontal
            pygame.draw.rect(surface, DARK_GRAY, (self.x, self.y, 90, 30))
            pygame.draw.circle(surface, RED if self.state == 'red' else GRAY, (self.x + 15, self.y + 15), 10)
            pygame.draw.circle(surface, YELLOW if self.state == 'yellow' else GRAY, (self.x + 45, self.y + 15), 10)
            pygame.draw.circle(surface, GREEN if self.state == 'green' else GRAY, (self.x + 75, self.y + 15), 10)

class Car(pygame.sprite.Sprite):
    """Represents a car in the simulation. Modified to track movement."""
    _id_counter = 0

    def __init__(self, x, y, direction, maneuver):
        super().__init__()
        self.id = Car._id_counter
        Car._id_counter += 1

        self.original_direction = direction
        self.direction = direction
        self.speed = random.uniform(100, 200)
        self.color = WHITE  # <-- All regular cars are white
        self.maneuver = maneuver
        self.is_turning = False
        
        # RL-specific properties to track if car is stuck
        self.is_moving = True
        self.last_position = None
        self.stuck_timer = 0

        self.image_vertical = pygame.Surface([20, 40])
        self.image_vertical.fill(self.color)
        self.image_horizontal = pygame.Surface([40, 20])
        self.image_horizontal.fill(self.color)

        self.image = self.image_vertical if direction in ['up', 'down'] else self.image_horizontal
        self.rect = self.image.get_rect(topleft=(x, y))
        self.last_position = self.rect.center

    def update(self, dt, vertical_light, horizontal_light, all_sprites):
        """Moves the car and handles logic."""
        # --- Store current position to check for movement later ---
        self.last_position = self.rect.center
        
        can_move = True
        
        # --- Collision Avoidance ---
        sensor_rect = self.rect.copy()
        sensor_distance = 30
        if self.direction == 'up': sensor_rect.y -= sensor_distance
        elif self.direction == 'down': sensor_rect.y += sensor_distance
        elif self.direction == 'right': sensor_rect.x += sensor_distance
        elif self.direction == 'left': sensor_rect.x -= sensor_distance

        for car in list(all_sprites):
            if car is not self and sensor_rect.colliderect(car.rect):
                can_move = False
                break

        # --- Traffic Light Stopping Logic ---
        light = vertical_light if self.original_direction in ['up', 'down'] else horizontal_light
        stop_line_up, stop_line_right = 480, 310

        if self.original_direction == 'up':
            if not self.is_turning and light.state != 'green' and self.rect.top < stop_line_up and self.rect.bottom > stop_line_up - 20:
                can_move = False
        elif self.original_direction == 'right':
            if not self.is_turning and light.state != 'green' and self.rect.right > stop_line_right and self.rect.left < stop_line_right + 20:
                can_move = False
        
        # --- Turning Logic ---
        if can_move and self.maneuver != 'straight' and not self.is_turning:
            turn_initiated = False
            new_direction = self.direction
            if self.maneuver == 'left' and self.original_direction == 'right' and self.rect.centerx >= 360:
                new_direction, turn_initiated = 'up', True
            elif self.maneuver == 'right' and self.original_direction == 'up' and self.rect.centery <= 420:
                new_direction, turn_initiated = 'right', True

            if turn_initiated:
                self.is_turning = True
                self.direction = new_direction
                old_center = self.rect.center
                self.image = self.image_vertical if new_direction in ['up', 'down'] else self.image_horizontal
                if new_direction == 'up': self.rect = self.image.get_rect(centery=old_center[1], centerx=370)
                elif new_direction == 'right': self.rect = self.image.get_rect(centerx=old_center[0], centery=420)
                else: self.rect = self.image.get_rect(center=old_center)

        # --- Movement ---
        if can_move:
            move_amount = self.speed * dt
            if self.direction == 'up': self.rect.y -= move_amount
            elif self.direction == 'down': self.rect.y += move_amount
            elif self.direction == 'left': self.rect.x -= move_amount
            elif self.direction == 'right': self.rect.x += move_amount

        # --- Update Movement Status for RL Reward ---
        if self.rect.center == self.last_position:
            self.stuck_timer += dt
            if self.stuck_timer > 0.5: # Considered stuck if not moved for 0.5s
                 self.is_moving = False
        else:
            self.stuck_timer = 0
            self.is_moving = True

        # --- Remove if off-screen ---
        if self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT or self.rect.right < 0 or self.rect.left > SCREEN_WIDTH:
            self.kill()

class EmergencyVehicle(Car):
    """Represents an emergency vehicle that is always red."""
    def __init__(self, x, y, direction, maneuver):
        super().__init__(x, y, direction, maneuver)
        self.speed = random.uniform(200, 250)
        self.color = RED  # Emergency vehicles are always red
        self.image_vertical.fill(self.color)
        self.image_horizontal.fill(self.color)

    def update(self, dt, vertical_light, horizontal_light, all_sprites):
        # No flashing, always red
        self.image_vertical.fill(self.color)
        self.image_horizontal.fill(self.color)
        self.image = self.image_vertical if self.direction in ['up', 'down'] else self.image_horizontal
        super().update(dt, vertical_light, horizontal_light, all_sprites)

# --- RL Environment ---
class TrafficEnv:
    """The Pygame simulation wrapped as a Reinforcement Learning Environment."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("RL Traffic Simulation")
        self.clock = pygame.time.Clock()
        self.min_switch_interval = 2.0  # seconds
        self.time_since_last_switch = 0.0
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.vertical_light = TrafficLight(460, 250, 'vertical', 'green')
        self.horizontal_light = TrafficLight(250, 460, 'horizontal', 'red')
        self.light_that_was_green = self.vertical_light
        self.is_switching = False
        self.switch_timer = 0
        self.yellow_duration = 1.0
        self.all_sprites = pygame.sprite.Group()
        self.car_spawn_timer = 0
        self.total_reward = 0
        self.time_since_last_switch = self.min_switch_interval  # allow immediate switch at start
        return self.get_state()

    def get_state(self):
        """Returns the current screen frame as a processed numpy array."""
        frame = pygame.surfarray.array3d(self.screen)  # (W, H, C)
        frame = np.transpose(frame, (1, 0, 2))        # (H, W, C)
        frame = tf.image.resize(frame, [IMG_HEIGHT, IMG_WIDTH])
        frame = tf.cast(frame, tf.float32) / 255.0    # Normalize to [0, 1]
        return frame  # shape: (IMG_HEIGHT, IMG_WIDTH, 3)

    def calculate_reward(self):
        reward = 0
        stuck_penalty = -5         # penalty for each stuck regular car
        emergency_bonus = 5.0        # bonus for moving emergency vehicle
        moving_bonus = 1           # bonus for each moving regular car
        emergency_stopped_penalty = -10  # big penalty if emergency vehicle is stopped

        if not self.all_sprites:
            return 0

        for car in self.all_sprites:
            if isinstance(car, EmergencyVehicle):
                if car.is_moving:
                    reward += emergency_bonus
                else:
                    reward += emergency_stopped_penalty
            else:
                if car.is_moving:
                    reward += moving_bonus
                else:
                    reward += stuck_penalty

        return reward

    def step(self, action):
        penalty = 0.0

        # --- Penalize switching too quickly ---
        if action == 1:
            if self.time_since_last_switch < self.min_switch_interval:
                penalty = -5.0  # You can tune this penalty value

        # --- Action Handling ---
        if action == 1 and not self.is_switching and self.time_since_last_switch >= self.min_switch_interval:
            self.is_switching = True
            self.light_that_was_green.state = 'yellow'
            self.switch_timer = self.yellow_duration
            self.time_since_last_switch = 0.0  # Reset timer on switch

        # --- Time Management ---
        for _ in range(10):
            dt = self.clock.tick(60) / 1000.0
            self.time_since_last_switch += dt

            # --- Light Switching Logic ---
            if self.is_switching:
                self.switch_timer -= dt
                if self.switch_timer <= 0:
                    self.is_switching = False
                    if self.light_that_was_green == self.vertical_light:
                        self.vertical_light.state = 'red'
                        self.horizontal_light.state = 'green'
                        self.light_that_was_green = self.horizontal_light
                    else:
                        self.horizontal_light.state = 'red'
                        self.vertical_light.state = 'green'
                        self.light_that_was_green = self.vertical_light

            # --- Car Spawning ---
            self.car_spawn_timer += dt
            if self.car_spawn_timer >= 1.0:
                self.car_spawn_timer -= 1.0
                self._spawn_car('south')
                self._spawn_car('west')
                
            # --- Update Sprites ---
            self.all_sprites.update(dt, self.vertical_light, self.horizontal_light, self.all_sprites)
            
            # --- Drawing ---
            self._draw_elements()

        # --- Calculate reward and get next state ---
        reward = self.calculate_reward() + penalty
        next_state = self.get_state()
        done = False

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        return next_state, reward, done

    def _spawn_car(self, spawn_point):
        """Helper function to spawn a new car."""
        if spawn_point == 'south':
            maneuver = random.choice(['straight', 'straight', 'right'])
            x = 410 if maneuver == 'right' else random.choice([370, 410])
            y, direction = SCREEN_HEIGHT, 'up'
        elif spawn_point == 'west':
            maneuver = random.choice(['straight', 'straight', 'left'])
            y = 370 if maneuver == 'left' else random.choice([370, 410])
            x, direction = -40, 'right'
        else: return

        if self._is_spawn_clear(x, y, direction):
            car_class = EmergencyVehicle if random.random() < 0.1 else Car
            car = car_class(x, y, direction, maneuver)
            self.all_sprites.add(car)

    def _is_spawn_clear(self, x, y, direction):
        """Check if the spawn area is clear for a new car."""
        spawn_rect = pygame.Rect(x, y, 40, 40) # A larger rect for safety
        for car in self.all_sprites:
            if spawn_rect.colliderect(car.rect):
                return False
        return True

    def _draw_elements(self):
        """Draws all simulation elements to the screen."""
        # Draw Roads
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, GRAY, (350, 0, 100, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, GRAY, (0, 350, SCREEN_WIDTH, 100))
        for y in range(0, 350, 40): pygame.draw.rect(self.screen, WHITE, (398, y, 4, 20))
        for y in range(450, SCREEN_HEIGHT, 40): pygame.draw.rect(self.screen, WHITE, (398, y, 4, 20))
        for x in range(0, 350, 40): pygame.draw.rect(self.screen, WHITE, (x, 398, 20, 4))
        for x in range(450, SCREEN_WIDTH, 40): pygame.draw.rect(self.screen, WHITE, (x, 398, 20, 4))
        
        # Draw Lights and Cars
        self.vertical_light.draw(self.screen)
        self.horizontal_light.draw(self.screen)
        self.all_sprites.draw(self.screen)
        pygame.display.flip()

# --- RL Agent ---
class DQNAgent:
    """A Deep Q-Network Agent to learn traffic control."""
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = collections.deque(maxlen=10000) # Increased replay buffer size
        
        # Hyperparameters
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        """Builds the CNN model."""
        model = models.Sequential()
        # Input shape: (height, width, channels)
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear')) # Output Q-values for each action
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.memory.append((state.numpy(), action, reward, next_state.numpy(), done))

    def act(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        act_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(act_values[0])  # Exploit

    def replay(self, batch_size):
        """Trains the network on a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        target_f = self.model.predict_on_batch(states)
        
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
            
        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --- Main Training Loop ---
if __name__ == "__main__":
    env = TrafficEnv()
    state_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    action_size = 2  # 0: do nothing, 1: switch light
    agent = DQNAgent(state_shape, action_size)

    episodes = 1000  # Increased for better convergence
    batch_size = 64  # Increased batch size

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        num_steps = 50

        for time_step in range(num_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # --- Show what the model sees (BGR image) ---
            frame = (state.numpy() * 255).astype(np.uint8) if hasattr(state, 'numpy') else (state * 255).astype(np.uint8)
            frame = frame[..., ::-1]  # RGB to BGR for correct colors
            cv2.imshow("Model Input (BGR)", frame)
            cv2.waitKey(1)  # 1 ms delay to update window

            # --- Print reward at each step ---
            print(f"Step {time_step+1}: reward={reward:.3f}")

            if done:
                break

        # Train once per episode after collecting experiences
        agent.replay(batch_size)

        avg_reward = total_reward / num_steps
        print(f"Episode {e+1}/{episodes}, Total Score: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # Optionally, save the model weights periodically
        if (e + 1) % 50 == 0:
            agent.model.save_weights(f'traffic_dqn_episode_{e+1}.weights.h5')

    cv2.destroyAllWindows()
    pygame.quit()