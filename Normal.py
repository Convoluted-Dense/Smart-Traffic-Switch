import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Traffic Light Simulation (Infinite Spawn)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
DARK_GRAY = (50, 50, 50)

# --- Classes ---
class TrafficLight:
    """Represents a traffic light at the intersection."""
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation  # 'vertical' or 'horizontal'
        self.state = 'red'
        self.timer = 0
        self.green_duration = 10
        self.yellow_duration = 3
        self.red_duration = self.green_duration + self.yellow_duration

        if orientation == 'vertical':
            self.state = 'green'  # Vertical starts green
        else:
            self.state = 'red'  # Horizontal starts red

    def update(self, dt):
        """Updates the traffic light's state based on the timer."""
        self.timer += dt
        if self.state == 'green' and self.timer >= self.green_duration:
            self.state = 'yellow'
            self.timer = 0
        elif self.state == 'yellow' and self.timer >= self.yellow_duration:
            self.state = 'red'
            self.timer = 0
        elif self.state == 'red' and self.timer >= self.red_duration:
            self.state = 'green'
            self.timer = 0

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
    """Represents a car in the simulation."""
    _id_counter = 0  # class-level counter for unique IDs

    def __init__(self, x, y, direction, maneuver):
        super().__init__()
        # assign a small unique id to each car for easier debug
        self.id = Car._id_counter
        Car._id_counter += 1

        self.original_direction = direction
        self.direction = direction
        self.speed = random.uniform(100, 200)  # Pixels per second
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.maneuver = maneuver
        self.is_turning = False

        # Surfaces for orientations
        self.image_vertical = pygame.Surface([20, 40])
        self.image_vertical.fill(self.color)
        self.image_horizontal = pygame.Surface([40, 20])
        self.image_horizontal.fill(self.color)

        self.image = self.image_vertical if direction in ['up', 'down'] else self.image_horizontal
        self.rect = self.image.get_rect(topleft=(x, y))

    def update(self, dt, vertical_light, horizontal_light, all_sprites):
        """Moves the car and handles traffic light, turning, and collision logic."""
        can_move = True

        # --- Collision Avoidance ---
        sensor_rect = self.rect.copy()
        sensor_distance = 30
        if self.direction == 'up':
            sensor_rect.y -= sensor_distance
        elif self.direction == 'down':
            sensor_rect.y += sensor_distance
        elif self.direction == 'right':
            sensor_rect.x += sensor_distance
        elif self.direction == 'left':
            sensor_rect.x -= sensor_distance

        # iterate over a snapshot of sprites to avoid surprises if group changes while iterating
        for car in list(all_sprites):
            if car is self:
                continue
            if sensor_rect.colliderect(car.rect):
                can_move = False
                break

        # --- Traffic Light Stopping Logic ---
        light = vertical_light if self.original_direction in ['up', 'down'] else horizontal_light
        stop_line_up = 480
        stop_line_down = 310
        stop_line_left = 480
        stop_line_right = 310

        stop_line = 0
        front_edge = 0

        if self.original_direction == 'up':
            stop_line = stop_line_up
            front_edge = self.rect.top
            if can_move and not self.is_turning and light.state != 'green' and stop_line + 5 >= front_edge >= stop_line - self.rect.height:
                can_move = False
        elif self.original_direction == 'right':
            stop_line = stop_line_right
            front_edge = self.rect.right
            if can_move and not self.is_turning and light.state != 'green' and stop_line - 5 <= front_edge <= stop_line + self.rect.width:
                can_move = False

        # --- Turning Logic ---
        if can_move and self.maneuver != 'straight' and not self.is_turning:
            turn_initiated = False
            new_direction = self.direction

            if self.maneuver == 'left':
                if self.original_direction == 'right' and self.rect.centerx >= 360:
                    new_direction = 'up'
                    turn_initiated = True
            elif self.maneuver == 'right':
                if self.original_direction == 'up' and self.rect.centery <= 420:
                    new_direction = 'right'
                    turn_initiated = True

            if turn_initiated:
                self.is_turning = True
                self.direction = new_direction
                old_center = self.rect.center
                self.image = self.image_vertical if new_direction in ['up', 'down'] else self.image_horizontal
                if new_direction == 'up':
                    self.rect = self.image.get_rect(centery=old_center[1], centerx=370)
                elif new_direction == 'right':
                    self.rect = self.image.get_rect(centerx=old_center[0], centery=420)
                else:
                    self.rect = self.image.get_rect(center=old_center)

        # --- Movement ---
        if can_move:
            move_amount = self.speed * dt
            if self.direction == 'up':
                self.rect.y -= move_amount
            elif self.direction == 'down':
                self.rect.y += move_amount
            elif self.direction == 'left':
                self.rect.x -= move_amount
            elif self.direction == 'right':
                self.rect.x += move_amount

        # Remove car if off-screen and log immediately after removal for debug
        if (self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT or
                self.rect.right < 0 or self.rect.left > SCREEN_WIDTH):
            print(f"Removing car id {self.id} at position {self.rect.topleft}, direction {self.direction}")
            self.kill()
            return  # return early after killing so rest of update isn't used

def draw_roads(surface):
    """Draws the intersection roads."""
    pygame.draw.rect(surface, GRAY, (350, 0, 100, SCREEN_HEIGHT))
    pygame.draw.rect(surface, GRAY, (0, 350, SCREEN_WIDTH, 100))
    for y in range(0, 350, 40):
        pygame.draw.rect(surface, WHITE, (398, y, 4, 20))
    for y in range(450, SCREEN_HEIGHT, 40):
        pygame.draw.rect(surface, WHITE, (398, y, 4, 20))
    for x in range(0, 350, 40):
        pygame.draw.rect(surface, WHITE, (x, 398, 20, 4))
    for x in range(450, SCREEN_WIDTH, 40):
        pygame.draw.rect(surface, WHITE, (x, 398, 20, 4))

# Add this before the main loop
last_spawn_time = {0: 0, 1: 0, 2: 0, 3: 0}  # one entry per spawn point
spawn_cooldown = 1000  # milliseconds (1 sec per spawn point)

def is_spawn_clear(all_sprites, x, y, direction):
    """Check if the spawn area is clear for a new car."""
    if direction == 'up':
        spawn_rect = pygame.Rect(x, y, 20, 40)
    elif direction == 'right':
        spawn_rect = pygame.Rect(x, y, 40, 20)
    else:
        return True  # fallback, should not happen

    for car in all_sprites:
        if spawn_rect.colliderect(car.rect):
            return False
    return True

def main():
    """Main game loop."""
    clock = pygame.time.Clock()
    running = True

    vertical_light = TrafficLight(460, 250, 'vertical')
    horizontal_light = TrafficLight(250, 460, 'horizontal')

    all_sprites = pygame.sprite.Group()
    car_spawn_timer = 0
    SPAWN_INTERVAL_MIN = 0.5
    SPAWN_INTERVAL_MAX = 2.0
    spawn_interval = random.uniform(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)

    last_time = time.time()
    spawn_count = 0  # Debug: Track number of spawn events

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Time Management ---
        # only call clock.tick once per frame (this returns milliseconds since last call)
        dt = clock.tick(60) / 1000.0

        # --- Game Logic ---
        vertical_light.update(dt)
        horizontal_light.update(dt)
        all_sprites.update(dt, vertical_light, horizontal_light, all_sprites)

        # --- Car Spawning: Only one car per spawn point per second, and only if clear ---
        car_spawn_timer += dt
        if car_spawn_timer >= 1.0:
            car_spawn_timer -= 1.0

            for spawn_point in ['south', 'west']:
                car = None
                if spawn_point == 'south':
                    maneuver = random.choice(['straight', 'straight', 'right'])
                    x = 410 if maneuver == 'right' else random.choice([370, 410])
                    y = SCREEN_HEIGHT
                    direction = 'up'
                elif spawn_point == 'west':
                    maneuver = random.choice(['straight', 'straight', 'left'])
                    y = 370 if maneuver == 'left' else random.choice([370, 410])
                    x = -40
                    direction = 'right'
                else:
                    continue

                # Only spawn if the area is clear
                if is_spawn_clear(all_sprites, x, y, direction):
                    car = Car(x, y, direction, maneuver)
                    all_sprites.add(car)
                    print(f"Added car id {car.id} from {spawn_point} maneuver {car.maneuver} at {car.rect.topleft}")

            print(f"Spawn event {spawn_count}: sprite count = {len(all_sprites)}")
            spawn_count += 1

        # --- Drawing ---
        screen.fill(BLACK)
        draw_roads(screen)
        vertical_light.draw(screen)
        horizontal_light.draw(screen)
        all_sprites.draw(screen)

        # --- Update Display ---
        pygame.display.flip()
        # (no second clock.tick here)

    pygame.quit()

if __name__ == "__main__":
    main()
