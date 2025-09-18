import argparse, random, math
import pygame as pg
import numpy as np
import os

# ==== Optional: import TensorFlow/Keras for LSTM ====
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ========== RL CONFIG ==========
ACTIONS = ["NS_green", "EW_green"]
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
Q = {}

# ========== SIMULATION CONFIG ==========
WIDTH, HEIGHT = 1000, 500
LANE_WIDTH = 40
GREEN_LIGHT_DURATION = 240
YELLOW_LIGHT_DURATION = 60

# ========== ASSET PATHS ==========
IMAGE_PATH = 'images/'
SIGNAL_PATH = os.path.join(IMAGE_PATH, 'Signals/')
VEHICLE_IMAGES_PATH = {}
DIRECTIONS = ['Up', 'Down', 'Left', 'Right']
for direction in DIRECTIONS:
    dir_path = os.path.join(IMAGE_PATH, direction)
    if os.path.exists(dir_path):
        VEHICLE_IMAGES_PATH[direction] = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    else:
        print(f"Warning: Directory not found at '{dir_path}'. Vehicles for this direction won't be loaded.")
        VEHICLE_IMAGES_PATH[direction] = []

# Traffic light positions for drawing
SIGNAL_POSITIONS = {
    "NS_top": (WIDTH // 2 - 110, HEIGHT // 2 - 200),
    "NS_bottom": (WIDTH // 2 + 70, HEIGHT // 2 + 110),
    "EW_left": (WIDTH // 2 - 200, HEIGHT // 2 + 70),
    "EW_right": (WIDTH // 2 + 110, HEIGHT // 2 - 110)
}

# ========== INITIALIZE PYGAME ==========
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("RL+LSTM Traffic Simulation")
clock = pg.time.Clock()


# ---------- Asset Loading ----------
def load_assets():
    """Loads and scales all images and returns them in a dictionary."""
    assets = {
        'background': pg.transform.scale(pg.image.load(os.path.join(IMAGE_PATH, 'mod_int.png')).convert(),
                                         (WIDTH, HEIGHT)),
        'signals': {
            'red': pg.transform.scale(pg.image.load(os.path.join(SIGNAL_PATH, 'red.png')).convert_alpha(), (40, 90)),
            'yellow': pg.transform.scale(pg.image.load(os.path.join(SIGNAL_PATH, 'yellow.png')).convert_alpha(),
                                         (40, 90)),
            'green': pg.transform.scale(pg.image.load(os.path.join(SIGNAL_PATH, 'green.png')).convert_alpha(),
                                        (40, 90)),
        },
        'vehicles': {'Up': [], 'Down': [], 'Left': [], 'Right': []}
    }
    for direction, paths in VEHICLE_IMAGES_PATH.items():
        for path in paths:
            img = pg.image.load(path).convert_alpha()
            if 'bus' in path or 'truck' in path:
                scale_factor = 0.8
            elif 'bike' in path:
                scale_factor = 1.3
            else:
                scale_factor = 1
            scaled_img = pg.transform.scale(img,
                                            (int(img.get_width() * scale_factor), int(img.get_height() * scale_factor)))
            assets['vehicles'][direction].append(scaled_img)
    return assets


# ---------- RL Utility ----------
def get_state(ns_count, ew_count):
    return (min(ns_count // 3, 4), min(ew_count // 3, 4))


def choose_action(state):
    if random.random() < EPSILON or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)


def update_q(state, action, reward, next_state):
    if state not in Q: Q[state] = {a: 0 for a in ACTIONS}
    if next_state not in Q: Q[next_state] = {a: 0 for a in ACTIONS}
    best_next = max(Q[next_state].values())
    Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])


# ---------- Vehicle Class ----------
class Vehicle:
    def __init__(self, direction, assets):
        self.direction = direction
        self.image = random.choice(assets['vehicles'][direction])
        self.rect = self.image.get_rect()
        self.speed = 2

        if direction == "Down":
            self.rect.x, self.rect.y = random.choice([WIDTH // 2 - 40, WIDTH // 2 - 85]), -self.rect.height
            self.vx, self.vy = 0, self.speed
            self.stop_pos = HEIGHT // 2 - 80
        elif direction == "Up":
            self.rect.x, self.rect.y = random.choice([WIDTH // 2 + 15, WIDTH // 2 + 60]), HEIGHT
            self.vx, self.vy = 0, -self.speed
            self.stop_pos = HEIGHT // 2 + 80
        elif direction == "Right":
            self.rect.x, self.rect.y = -self.rect.width, random.choice([HEIGHT // 2 - 40, HEIGHT // 2 - 85])
            self.vx, self.vy = self.speed, 0
            self.stop_pos = WIDTH // 2 - 80
        else:
            self.rect.x, self.rect.y = WIDTH, random.choice([HEIGHT // 2 + 15, HEIGHT // 2 + 60])
            self.vx, self.vy = -self.speed, 0
            self.stop_pos = WIDTH // 2 + 80

    def update(self, light_status, all_vehicles):
        is_red_light = (self.direction in ("Up", "Down") and light_status in ("EW_green", "EW_yellow")) or \
                       (self.direction in ("Left", "Right") and light_status in ("NS_green", "NS_yellow"))

        should_stop = False
        if is_red_light:
            if self.direction == "Down" and self.rect.bottom >= self.stop_pos:
                should_stop = True
            elif self.direction == "Up" and self.rect.top <= self.stop_pos:
                should_stop = True
            elif self.direction == "Right" and self.rect.right >= self.stop_pos:
                should_stop = True
            elif self.direction == "Left" and self.rect.left <= self.stop_pos:
                should_stop = True

        for other in all_vehicles:
            if other is self or self.direction != other.direction: continue

            is_ahead = False
            if self.direction in ("Up", "Down") and abs(self.rect.x - other.rect.x) < 10:
                if self.direction == "Down" and other.rect.top > self.rect.bottom:
                    is_ahead = True
                elif self.direction == "Up" and other.rect.bottom < self.rect.top:
                    is_ahead = True
            elif self.direction in ("Left", "Right") and abs(self.rect.y - other.rect.y) < 10:
                if self.direction == "Right" and other.rect.left > self.rect.right:
                    is_ahead = True
                elif self.direction == "Left" and other.rect.right < self.rect.left:
                    is_ahead = True

            if is_ahead:
                distance = math.hypot(self.rect.centerx - other.rect.centerx, self.rect.centery - other.rect.centery)
                # MODIFIED: Increased follow distance slightly for better spacing
                if distance < self.rect.height * 1.8 + 5:
                    should_stop = True
                    break

        if not should_stop:
            self.rect.x += self.vx
            self.rect.y += self.vy

    def draw(self, surf):
        surf.blit(self.image, self.rect)


# ---------- Main Simulation ----------
def run_simulation(episodes=5):
    assets = load_assets()
    vehicles = []

    light_status = "NS_green"
    phase_timer = GREEN_LIGHT_DURATION

    # MODIFIED: Define spawn zones to check for collisions before adding a new car
    SPAWN_ZONES = {
        "Down": [pg.Rect(WIDTH // 2 - 85, -100, 50, 100), pg.Rect(WIDTH // 2 - 40, -100, 50, 100)],
        "Up": [pg.Rect(WIDTH // 2 + 15, HEIGHT, 50, 100), pg.Rect(WIDTH // 2 + 60, HEIGHT, 50, 100)],
        "Right": [pg.Rect(-100, HEIGHT // 2 - 85, 100, 50), pg.Rect(-100, HEIGHT // 2 - 40, 100, 50)],
        "Left": [pg.Rect(WIDTH, HEIGHT // 2 + 15, 100, 50), pg.Rect(WIDTH, HEIGHT // 2 + 60, 100, 50)]
    }

    for ep in range(episodes):
        running = True
        frame = 0
        while running:
            for e in pg.event.get():
                if e.type == pg.QUIT: pg.quit(); return

            # MODIFIED: Replaced the old spawn logic with a new one that checks the spawn zone
            if random.random() < 0.05:  # Slightly increased spawn rate
                direction = random.choice(DIRECTIONS)
                if assets['vehicles'][direction]:
                    # Check if the spawn zones for the chosen direction are clear
                    spawn_area_clear = True
                    for zone in SPAWN_ZONES[direction]:
                        for v in vehicles:
                            if zone.colliderect(v.rect):
                                spawn_area_clear = False
                                break
                        if not spawn_area_clear:
                            break

                    if spawn_area_clear:
                        vehicles.append(Vehicle(direction, assets))
            # --- END OF MODIFIED SECTION ---

            phase_timer -= 1
            if phase_timer <= 0:
                if light_status == "NS_green":
                    light_status = "NS_yellow"
                    phase_timer = YELLOW_LIGHT_DURATION
                elif light_status == "NS_yellow":
                    light_status = "EW_green"
                    phase_timer = GREEN_LIGHT_DURATION
                elif light_status == "EW_green":
                    light_status = "EW_yellow"
                    phase_timer = YELLOW_LIGHT_DURATION
                elif light_status == "EW_yellow":
                    light_status = "NS_green"
                    phase_timer = GREEN_LIGHT_DURATION

            for v in vehicles: v.update(light_status, vehicles)

            vehicles = [v for v in vehicles if screen.get_rect().colliderect(v.rect)]

            screen.blit(assets['background'], (0, 0))

            if light_status == "NS_green":
                screen.blit(assets['signals']['green'], SIGNAL_POSITIONS["NS_top"]);
                screen.blit(assets['signals']['green'], SIGNAL_POSITIONS["NS_bottom"])
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["EW_left"]);
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["EW_right"])
            elif light_status == "NS_yellow":
                screen.blit(assets['signals']['yellow'], SIGNAL_POSITIONS["NS_top"]);
                screen.blit(assets['signals']['yellow'], SIGNAL_POSITIONS["NS_bottom"])
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["EW_left"]);
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["EW_right"])
            elif light_status == "EW_green":
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["NS_top"]);
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["NS_bottom"])
                screen.blit(assets['signals']['green'], SIGNAL_POSITIONS["EW_left"]);
                screen.blit(assets['signals']['green'], SIGNAL_POSITIONS["EW_right"])
            elif light_status == "EW_yellow":
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["NS_top"]);
                screen.blit(assets['signals']['red'], SIGNAL_POSITIONS["NS_bottom"])
                screen.blit(assets['signals']['yellow'], SIGNAL_POSITIONS["EW_left"]);
                screen.blit(assets['signals']['yellow'], SIGNAL_POSITIONS["EW_right"])

            for v in vehicles: v.draw(screen)

            pg.display.flip()
            clock.tick(60)
            frame += 1
            if frame > 1500: running = False
    print("Simulation finished.")


# ---------- Main Entry ----------
if __name__ == "__main__":
    run_simulation(episodes=1)
    pg.quit()