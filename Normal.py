import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Traffic Light Simulation (Fixed Edition v3)")

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
    def __init__(self, x, y, direction, maneuver):
        super().__init__()
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
        sensor_distance = 30  # Adjusted for better detection
        if self.direction == 'up':
            sensor_rect.y -= sensor_distance
        elif self.direction == 'down':
            sensor_rect.y += sensor_distance
        elif self.direction == 'right':
            sensor_rect.x += sensor_distance
        elif self.direction == 'left':
            sensor_rect.x -= sensor_distance

        for car in all_sprites:
            if car != self and sensor_rect.colliderect(car.rect):
                can_move = False
                break

        # --- Traffic Light Stopping Logic ---
        light = vertical_light if self.original_direction == 'up' else horizontal_light
        stop_line = 480 if self.original_direction == 'up' else 340  # Adjusted stop line for westbound
        front_edge = self.rect.bottom if self.original_direction == 'up' else self.rect.right

        if can_move and not self.is_turning:
            if light.state != 'green' and front_edge >= stop_line - self.speed * dt and front_edge <= stop_line + 10:
                can_move = False

        # --- Turning Logic ---
        if can_move and self.maneuver != 'straight' and not self.is_turning:
            turn_initiated = False
            new_direction = self.direction

            if self.maneuver == 'left':
                # From west (right/eastbound) turning north (up)
                if self.original_direction == 'right' and self.rect.centerx >= 360:  # Earlier turn trigger
                    new_direction = 'up'
                    turn_initiated = True
            elif self.maneuver == 'right':
                # From south (up/northbound) turning east (right)
                if self.original_direction == 'up' and self.rect.centery <= 420:
                    new_direction = 'right'
                    turn_initiated = True

            if turn_initiated:
                self.is_turning = True
                self.direction = new_direction
                old_center = self.rect.center
                self.image = self.image_vertical if new_direction in ['up', 'down'] else self.image_horizontal
                # Adjust position to land in correct lane
                if new_direction == 'up':
                    self.rect = self.image.get_rect(centery=old_center[1], centerx=370)  # Left lane
                elif new_direction == 'right':
                    self.rect = self.image.get_rect(centerx=old_center[0], centery=420)  # Bottom lane
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

        # Remove car if off-screen
        if (self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT or
            self.rect.right < 0 or self.rect.left > SCREEN_WIDTH):
            self.kill()

def draw_roads(surface):
    """Draws the intersection roads."""
    # Vertical road (one-way northbound)
    pygame.draw.rect(surface, GRAY, (350, 0, 100, SCREEN_HEIGHT))
    # Horizontal road (one-way eastbound)
    pygame.draw.rect(surface, GRAY, (0, 350, SCREEN_WIDTH, 100))
    # Center lines, limited to arms outside intersection
    # Vertical road: north arm (y=0 to 350)
    for y in range(0, 350, 40):
        pygame.draw.rect(surface, WHITE, (398, y, 4, 20))
    # Vertical road: south arm (y=450 to 800)
    for y in range(450, SCREEN_HEIGHT, 40):
        pygame.draw.rect(surface, WHITE, (398, y, 4, 20))
    # Horizontal road: west arm (x=0 to 350)
    for x in range(0, 350, 40):
        pygame.draw.rect(surface, WHITE, (x, 398, 20, 4))
    # Horizontal road: east arm (x=450 to 800)
    for x in range(450, SCREEN_WIDTH, 40):
        pygame.draw.rect(surface, WHITE, (x, 398, 20, 4))

def main():
    """Main game loop."""
    clock = pygame.time.Clock()
    running = True

    vertical_light = TrafficLight(460, 250, 'vertical')
    horizontal_light = TrafficLight(250, 460, 'horizontal')

    all_sprites = pygame.sprite.Group()
    car_spawn_timer = 0

    last_time = time.time()

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Time Management ---
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # --- Game Logic ---
        vertical_light.update(dt)
        horizontal_light.update(dt)
        all_sprites.update(dt, vertical_light, horizontal_light, all_sprites)

        # Spawn new cars from south and west only
        car_spawn_timer += dt
        if car_spawn_timer > 1.5:
            car_spawn_timer = 0
            spawn_point = random.choice(['south', 'west'])

            car = None
            if spawn_point == 'south':
                maneuver = random.choice(['straight', 'straight', 'right'])  # Favor straight, legal only
                if maneuver == 'right':
                    x = 410  # Right lane for right turn
                else:
                    x = random.choice([370, 410])  # Either lane for straight
                car = Car(x, SCREEN_HEIGHT, 'up', maneuver)
            elif spawn_point == 'west':
                maneuver = random.choice(['straight', 'straight', 'left'])  # Favor straight, legal only
                if maneuver == 'left':
                    y = 370  # Top lane for left turn
                else:
                    y = random.choice([370, 410])  # Either lane for straight
                car = Car(-40, y, 'right', maneuver)

            if car:
                all_sprites.add(car)

        # --- Drawing ---
        screen.fill(BLACK)
        draw_roads(screen)
        vertical_light.draw(screen)
        horizontal_light.draw(screen)
        all_sprites.draw(screen)

        # --- Update Display ---
        pygame.display.flip()

        # --- Frame Rate ---
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()