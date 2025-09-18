# traffic_rl_lstm_sim_improved.py
import argparse
import os
import random
import math
import time
from collections import deque
import csv

import numpy as np
import pygame as pg

# Optional TF/Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 1000, 600
FPS = 60

IMAGE_PATH = "images"
SIGNAL_PATH = os.path.join(IMAGE_PATH, "Signals")
DIRECTIONS = ["Up", "Down", "Left", "Right"]

# Improved lane positions based on intersection layout
LANE_POS = {
    "Down": [WIDTH // 2 - 35, WIDTH // 2 - 70],  # Right lanes for Down traffic
    "Up": [WIDTH // 2 + 35, WIDTH // 2 + 70],  # Left lanes for Up traffic
    "Right": [HEIGHT // 2 - 35, HEIGHT // 2 - 70],  # Bottom lanes for Right traffic
    "Left": [HEIGHT // 2 + 35, HEIGHT // 2 + 70],  # Top lanes for Left traffic
}

# Spawn positions (back to off-screen but close)
SPAWN_POS = {
    "Down": {"x_lanes": [WIDTH // 2 - 35, WIDTH // 2 - 70], "y": -80},
    "Up": {"x_lanes": [WIDTH // 2 + 35, WIDTH // 2 + 70], "y": HEIGHT + 80},
    "Right": {"x": -80, "y_lanes": [HEIGHT // 2 - 35, HEIGHT // 2 - 70]},
    "Left": {"x": WIDTH + 80, "y_lanes": [HEIGHT // 2 + 35, HEIGHT // 2 + 70]},
}

# Stop lines before intersection
STOP_LINES = {
    "Down": HEIGHT // 2 - 120,
    "Up": HEIGHT // 2 + 120,
    "Right": WIDTH // 2 - 120,
    "Left": WIDTH // 2 + 120
}

# Traffic light positions
LIGHT_POSITIONS = {
    "NS_top": (WIDTH // 2 - 100, HEIGHT // 2 - 180),
    "NS_bottom": (WIDTH // 2 + 60, HEIGHT // 2 + 120),
    "EW_left": (WIDTH // 2 - 180, HEIGHT // 2 + 60),
    "EW_right": (WIDTH // 2 + 120, HEIGHT // 2 - 100)
}

# RL action set and timing
ACTIONS = ["NS_short", "NS_long", "EW_short", "EW_long"]
GREEN_DURATIONS = {"short": 5.0, "long": 10.0}  # seconds
YELLOW_DURATION = 3.0  # seconds
ALL_RED_DURATION = 1.0  # seconds

# Q-learning hyperparams
ALPHA = 0.15
GAMMA = 0.95
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 0.995

Q_TABLE_PATH = "q_table.npy"
COUNTS_CSV = "vehicle_counts.csv"
LSTM_MODEL_PATH = "lstm_model.h5"

# ---------------- Pygame init ---------------
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Smart Traffic Signal - RL + LSTM")
clock = pg.time.Clock()
font = pg.font.SysFont(None, 24)
small_font = pg.font.SysFont(None, 18)


# ---------------- Asset loading ----------------
def load_assets():
    assets = {
        "background": None,
        "signals": {},
        "vehicles": {d: [] for d in DIRECTIONS}
    }

    # Background
    bg_path = os.path.join(IMAGE_PATH, "mod_int.png")
    if os.path.exists(bg_path):
        bg = pg.image.load(bg_path).convert()
        assets["background"] = pg.transform.scale(bg, (WIDTH, HEIGHT))

    # Traffic light signals
    for name in ("red", "yellow", "green"):
        p = os.path.join(SIGNAL_PATH, f"{name}.png")
        if os.path.exists(p):
            img = pg.image.load(p).convert_alpha()
            assets["signals"][name] = pg.transform.smoothscale(img, (40, 90))

    # Vehicle images
    for d in DIRECTIONS:
        dpath = os.path.join(IMAGE_PATH, d)
        if os.path.isdir(dpath):
            for fname in sorted(os.listdir(dpath)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full = os.path.join(dpath, fname)
                    try:
                        img = pg.image.load(full).convert_alpha()
                        # Scale vehicles appropriately based on direction
                        if d in ["Up", "Down"]:
                            img = pg.transform.smoothscale(img, (40, 60))
                        else:  # Left, Right
                            img = pg.transform.smoothscale(img, (60, 40))
                        assets["vehicles"][d].append(img)
                        print(f"Loaded vehicle image: {fname} for direction {d}")
                    except Exception as e:
                        print(f"Warning loading {full}: {e}")

    return assets


# ---------------- LSTM Predictor (Fixed) ----------------
class LSTMPredictor:
    def __init__(self, lookback=10):
        if not TF_AVAILABLE:
            self.available = False
            return
        self.available = True
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.trained = False

    def build(self):
        if not self.available:
            return None
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model

    def prepare_data(self, data):
        if len(data) < self.lookback + 1:
            return None, None

        scaled_data = self.scaler.fit_transform(np.array(data).reshape(-1, 1))
        X, y = [], []

        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def train(self, traffic_data, epochs=50):
        if not self.available or len(traffic_data) < self.lookback + 10:
            return False

        X, y = self.prepare_data(traffic_data)
        if X is None:
            return False

        X = X.reshape((X.shape[0], X.shape[1], 1))

        if self.model is None:
            self.build()

        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=1, validation_split=0.2)
        self.trained = True

        # Save model
        try:
            self.model.save(LSTM_MODEL_PATH)
        except:
            pass

        return True

    def predict_next(self, recent_data):
        if not self.available or not self.trained or len(recent_data) < self.lookback:
            return 0

        try:
            # Use the last lookback points
            input_data = recent_data[-self.lookback:]
            scaled_input = self.scaler.transform(np.array(input_data).reshape(-1, 1))
            X = scaled_input.reshape((1, self.lookback, 1))

            prediction = self.model.predict(X, verbose=0)
            predicted_value = self.scaler.inverse_transform(prediction)[0, 0]
            return max(0, predicted_value)  # Ensure non-negative
        except:
            return 0


# ---------------- Q-Learning Agent (Improved) ----------------
class QAgent:
    def __init__(self, actions=ACTIONS, eps=EPS_START):
        self.actions = actions
        self.n_actions = len(actions)
        self.q_table = {}
        self.eps = eps
        self.learning_rate = ALPHA
        self.discount = GAMMA

    def get_state_key(self, state):
        """Convert state tuple to string key"""
        return str(state)

    def get_q_values(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def choose_action(self, state):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)

        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_values(state)
        next_q = self.get_q_values(next_state)

        current_q[action] += self.learning_rate * (
                reward + self.discount * np.max(next_q) - current_q[action]
        )

    def decay_epsilon(self):
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    def save(self, path=Q_TABLE_PATH):
        np.save(path, dict(self.q_table))

    def load(self, path=Q_TABLE_PATH):
        if os.path.exists(path):
            self.q_table = np.load(path, allow_pickle=True).item()
            return True
        return False


# ---------------- Vehicle Class (Fixed Lane Alignment) ----------------
class Vehicle:
    def __init__(self, direction, assets):
        self.direction = direction
        self.lane = random.choice([0, 1])  # Choose lane 0 or 1
        self.speed = random.uniform(1.5, 2.5)
        self.max_speed = self.speed
        self.stopped = False
        self.waiting_time = 0

        # Load vehicle image
        imgs = assets["vehicles"].get(direction, [])
        if imgs:
            self.image = random.choice(imgs)
            self.rect = self.image.get_rect()
            print(f"Loaded image for {direction} vehicle: {self.image.get_size()}")
        else:
            # Fallback rectangle
            print(f"No images found for {direction}, using fallback rectangle")
            if direction in ["Up", "Down"]:
                self.image = None
                self.rect = pg.Rect(0, 0, 40, 60)
            else:
                self.image = None
                self.rect = pg.Rect(0, 0, 60, 40)

        # Set initial position based on direction and lane
        self._set_spawn_position()

        # Set movement vector
        if direction == "Down":
            self.velocity = (0, self.speed)
            self.stop_line = STOP_LINES["Down"]
        elif direction == "Up":
            self.velocity = (0, -self.speed)
            self.stop_line = STOP_LINES["Up"]
        elif direction == "Right":
            self.velocity = (self.speed, 0)
            self.stop_line = STOP_LINES["Right"]
        else:  # Left
            self.velocity = (-self.speed, 0)
            self.stop_line = STOP_LINES["Left"]

        print(f"Created {direction} vehicle at ({self.rect.x}, {self.rect.y}) with velocity {self.velocity}")

    def _set_spawn_position(self):
        spawn = SPAWN_POS[self.direction]

        if self.direction == "Down":
            self.rect.centerx = spawn["x_lanes"][self.lane]
            self.rect.y = spawn["y"]
        elif self.direction == "Up":
            self.rect.centerx = spawn["x_lanes"][self.lane]
            self.rect.y = spawn["y"]
        elif self.direction == "Right":
            self.rect.x = spawn["x"]
            self.rect.centery = spawn["y_lanes"][self.lane]
        else:  # Left
            self.rect.x = spawn["x"]
            self.rect.centery = spawn["y_lanes"][self.lane]

    def should_stop_for_light(self, light_state):
        """Check if vehicle should stop for traffic light"""
        if light_state == "NS_GREEN":
            return self.direction in ["Left", "Right"]  # EW traffic must stop
        elif light_state == "EW_GREEN":
            return self.direction in ["Up", "Down"]  # NS traffic must stop
        elif light_state in ["NS_YELLOW", "EW_YELLOW", "ALL_RED"]:
            return True  # Everyone stops for yellow and all red
        else:
            return False  # Default: don't stop

    def check_collision_ahead(self, vehicles):
        """Check for vehicles ahead in the same lane"""
        safe_distance = 80  # Increased safe distance

        for other in vehicles:
            if other == self or other.direction != self.direction:
                continue

            # Check if in same lane (more lenient check)
            if self.direction in ["Up", "Down"]:
                if abs(self.rect.centerx - other.rect.centerx) > 40:  # More lenient
                    continue
            else:
                if abs(self.rect.centery - other.rect.centery) > 40:  # More lenient
                    continue

            # Check if other vehicle is ahead and close
            if self.direction == "Down":
                if (other.rect.centery > self.rect.centery and
                        other.rect.top - self.rect.bottom < safe_distance and
                        other.rect.top - self.rect.bottom > -20):  # Allow some overlap
                    return True
            elif self.direction == "Up":
                if (other.rect.centery < self.rect.centery and
                        self.rect.top - other.rect.bottom < safe_distance and
                        self.rect.top - other.rect.bottom > -20):
                    return True
            elif self.direction == "Right":
                if (other.rect.centerx > self.rect.centerx and
                        other.rect.left - self.rect.right < safe_distance and
                        other.rect.left - self.rect.right > -20):
                    return True
            elif self.direction == "Left":
                if (other.rect.centerx < self.rect.centerx and
                        self.rect.left - other.rect.right < safe_distance and
                        self.rect.left - other.rect.right > -20):
                    return True

        return False

    def at_stop_line(self):
        """Check if vehicle has passed the stop line"""
        margin = 10  # Small margin for stop line detection
        if self.direction == "Down":
            return self.rect.bottom >= (self.stop_line + margin)
        elif self.direction == "Up":
            return self.rect.top <= (self.stop_line - margin)
        elif self.direction == "Right":
            return self.rect.right >= (self.stop_line + margin)
        elif self.direction == "Left":
            return self.rect.left <= (self.stop_line - margin)
        return False

    def update(self, light_state, vehicles, dt):
        """Update vehicle position and state"""
        # Check if should stop for traffic light (only if haven't passed stop line)
        should_stop_for_light = (self.should_stop_for_light(light_state) and
                                 not self.at_stop_line())

        # Check for vehicles ahead (simplified collision detection)
        should_stop_for_collision = self.check_collision_ahead(vehicles)

        # Combine stopping conditions
        should_stop = should_stop_for_light or should_stop_for_collision

        # Update stopped state and waiting time
        if should_stop:
            self.stopped = True
            self.waiting_time += dt
        else:
            self.stopped = False
            self.waiting_time = 0

            # Move vehicle
            self.rect.x += int(self.velocity[0])
            self.rect.y += int(self.velocity[1])

        # Debug output occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 60 == 0:  # Every second at 60 FPS
            print(f"{self.direction} vehicle: pos=({self.rect.x},{self.rect.y}), "
                  f"light_stop={should_stop_for_light}, collision_stop={should_stop_for_collision}, "
                  f"stopped={self.stopped}, light_state={light_state}")

    def draw(self, surface):
        if self.image:
            # Use the actual vehicle image
            surface.blit(self.image, self.rect)
        else:
            # Fallback only if no image loaded
            colors = {
                "Up": (255, 100, 100),
                "Down": (100, 255, 100),
                "Left": (100, 100, 255),
                "Right": (255, 255, 100)
            }
            color = colors.get(self.direction, (255, 255, 255))
            pg.draw.rect(surface, color, self.rect)
            pg.draw.rect(surface, (0, 0, 0), self.rect, 2)

            # Draw direction indicator only for fallback
            font = pg.font.SysFont(None, 16)
            text = font.render(self.direction[0], True, (0, 0, 0))
            text_rect = text.get_rect(center=self.rect.center)
            surface.blit(text, text_rect)

    def is_off_screen(self):
        """Check if vehicle has left the screen"""
        margin = 50
        return (self.rect.right < -margin or self.rect.left > WIDTH + margin or
                self.rect.bottom < -margin or self.rect.top > HEIGHT + margin)


# ---------------- Traffic Controller ----------------
class TrafficController:
    def __init__(self, use_lstm=False):
        self.current_phase = "ALL_RED"
        self.phase_timer = 0
        self.yellow_timer = 0
        self.all_red_timer = 0

        # Data collection
        self.traffic_history = deque(maxlen=1000)
        self.waiting_history = deque(maxlen=100)

        # AI components
        self.q_agent = QAgent()
        self.lstm_predictor = None
        self.use_lstm = use_lstm

        if use_lstm and TF_AVAILABLE:
            self.lstm_predictor = LSTMPredictor()
            self._load_or_train_lstm()

        # Load Q-table if exists
        self.q_agent.load()

    def _load_or_train_lstm(self):
        """Load existing LSTM model or train a new one"""
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                from tensorflow.keras.models import load_model
                self.lstm_predictor.model = load_model(LSTM_MODEL_PATH)
                self.lstm_predictor.trained = True
                print("Loaded existing LSTM model")
            except:
                print("Failed to load LSTM model, will train new one")

        # Train if we have data
        if os.path.exists(COUNTS_CSV):
            try:
                import pandas as pd
                df = pd.read_csv(COUNTS_CSV)
                if len(df) > 50:
                    waiting_counts = df['total_waiting'].values
                    if self.lstm_predictor.train(waiting_counts):
                        print("LSTM model trained successfully")
            except Exception as e:
                print(f"Failed to train LSTM: {e}")

    def get_state(self, ns_waiting, ew_waiting, predicted_waiting=0):
        """Convert traffic counts to discrete state"""
        # Discretize waiting counts (0-4+ scale)
        ns_state = min(4, ns_waiting // 2)
        ew_state = min(4, ew_waiting // 2)

        if self.use_lstm and predicted_waiting > 0:
            pred_state = min(4, int(predicted_waiting) // 2)
            return (ns_state, ew_state, pred_state)

        return (ns_state, ew_state)

    def count_waiting_vehicles(self, vehicles):
        """Count vehicles waiting at each direction"""
        ns_waiting = sum(1 for v in vehicles if v.direction in ["Up", "Down"] and v.stopped)
        ew_waiting = sum(1 for v in vehicles if v.direction in ["Left", "Right"] and v.stopped)
        return ns_waiting, ew_waiting

    def calculate_reward(self, vehicles):
        """Calculate reward based on waiting times and queue lengths"""
        total_waiting = sum(1 for v in vehicles if v.stopped)
        total_waiting_time = sum(v.waiting_time for v in vehicles if v.stopped)

        # Negative reward for waiting (encourages reducing wait times)
        reward = -total_waiting - (total_waiting_time * 0.1)
        return reward

    def update(self, vehicles, dt, mode="play"):
        """Update traffic controller state"""
        ns_waiting, ew_waiting = self.count_waiting_vehicles(vehicles)
        total_waiting = ns_waiting + ew_waiting

        # Record data for LSTM
        self.waiting_history.append(total_waiting)

        # Get LSTM prediction if available
        predicted_waiting = 0
        if self.use_lstm and self.lstm_predictor and len(self.waiting_history) >= 10:
            predicted_waiting = self.lstm_predictor.predict_next(list(self.waiting_history))

        # Handle phase transitions
        if self.phase_timer <= 0:
            if self.current_phase == "ALL_RED":
                # Time to choose new action
                state = self.get_state(ns_waiting, ew_waiting, predicted_waiting)
                action_idx = self.q_agent.choose_action(state)
                action = ACTIONS[action_idx]

                # Store for Q-learning update
                self.last_state = state
                self.last_action = action_idx
                self.accumulated_reward = 0

                # Execute action
                if action.startswith("NS"):
                    self.current_phase = "NS_GREEN"
                    duration = GREEN_DURATIONS["long"] if "long" in action else GREEN_DURATIONS["short"]
                else:
                    self.current_phase = "EW_GREEN"
                    duration = GREEN_DURATIONS["long"] if "long" in action else GREEN_DURATIONS["short"]

                self.phase_timer = duration

            elif self.current_phase in ["NS_GREEN", "EW_GREEN"]:
                # Switch to yellow
                if self.current_phase == "NS_GREEN":
                    self.current_phase = "NS_YELLOW"
                else:
                    self.current_phase = "EW_YELLOW"
                self.phase_timer = YELLOW_DURATION

            elif self.current_phase in ["NS_YELLOW", "EW_YELLOW"]:
                # Switch to all red, then prepare for Q-learning update
                self.current_phase = "ALL_RED"
                self.phase_timer = ALL_RED_DURATION

                # Q-learning update
                if mode == "train":
                    next_state = self.get_state(ns_waiting, ew_waiting, predicted_waiting)
                    reward = self.calculate_reward(vehicles)
                    self.accumulated_reward += reward

                    self.q_agent.update_q_value(
                        self.last_state,
                        self.last_action,
                        self.accumulated_reward,
                        next_state
                    )

        # Update timers
        self.phase_timer -= dt

        # Accumulate reward for current action
        if hasattr(self, 'accumulated_reward'):
            self.accumulated_reward += self.calculate_reward(vehicles) * dt

    def get_light_color(self, direction_group):
        """Get the current light color for a direction group (NS or EW)"""
        if self.current_phase == "ALL_RED":
            return "red"
        elif self.current_phase == "NS_GREEN" and direction_group == "NS":
            return "green"
        elif self.current_phase == "EW_GREEN" and direction_group == "EW":
            return "green"
        elif self.current_phase == "NS_YELLOW" and direction_group == "NS":
            return "yellow"
        elif self.current_phase == "EW_YELLOW" and direction_group == "EW":
            return "yellow"
        else:
            return "red"

    def save_data(self):
        """Save traffic data to CSV"""
        try:
            with open(COUNTS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:  # File is empty, write header
                    writer.writerow(['timestamp', 'total_waiting', 'ns_waiting', 'ew_waiting'])

                # Write recent data
                if self.waiting_history:
                    writer.writerow([time.time(), self.waiting_history[-1], 0, 0])  # Simplified for now
        except Exception as e:
            print(f"Error saving data: {e}")


# ---------------- Main Simulation ----------------
def run_simulation(mode="play", episodes=50, display=True, use_lstm=False):
    global screen, clock

    assets = load_assets()
    controller = TrafficController(use_lstm=use_lstm)
    vehicles = []

    stats = {
        'total_vehicles': 0,
        'avg_wait_time': 0,
        'episode_rewards': []
    }

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")

        # Reset episode
        vehicles.clear()
        frame_count = 0
        episode_reward = 0
        spawn_timer = 0

        # Episode loop
        running = True
        episode_start = time.time()

        while running and frame_count < 3600:  # Max 60 seconds per episode
            dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return stats
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        # Manual spawn
                        direction = random.choice(DIRECTIONS)
                        vehicles.append(Vehicle(direction, assets))

            # Spawn vehicles less frequently to reduce crowding
            spawn_timer += dt
            if spawn_timer > random.uniform(1.0, 3.0):  # Back to less frequent spawning
                if len(vehicles) < 15:  # Reduced limit to prevent overcrowding
                    direction = random.choice(DIRECTIONS)
                    new_vehicle = Vehicle(direction, assets)

                    # Check spawn area is reasonably clear
                    can_spawn = True
                    for existing in vehicles:
                        if (abs(new_vehicle.rect.centerx - existing.rect.centerx) < 50 and
                                abs(new_vehicle.rect.centery - existing.rect.centery) < 50):
                            can_spawn = False
                            break

                    if can_spawn:
                        vehicles.append(new_vehicle)
                        stats['total_vehicles'] += 1
                        print(f"Spawned {direction} vehicle at ({new_vehicle.rect.x}, {new_vehicle.rect.y})")

                spawn_timer = 0

            # Update traffic controller
            controller.update(vehicles, dt, mode)

            # Update vehicles with less verbose logging
            for i, vehicle in enumerate(vehicles[:]):
                old_pos = (vehicle.rect.x, vehicle.rect.y)
                vehicle.update(controller.current_phase, vehicles, dt)
                new_pos = (vehicle.rect.x, vehicle.rect.y)

                if vehicle.is_off_screen():
                    vehicles.remove(vehicle)
                    print(f"Removed {vehicle.direction} vehicle (off-screen)")

            # Calculate episode reward
            episode_reward += controller.calculate_reward(vehicles) * dt

            # Render
            if display:
                # Clear screen
                if assets["background"]:
                    screen.blit(assets["background"], (0, 0))
                else:
                    screen.fill((50, 50, 50))

                # Draw traffic lights
                for light_name, pos in LIGHT_POSITIONS.items():
                    if "NS" in light_name:
                        color = controller.get_light_color("NS")
                    else:
                        color = controller.get_light_color("EW")

                    if color in assets["signals"]:
                        screen.blit(assets["signals"][color], pos)

                # Draw vehicles using their actual images
                vehicles_drawn = 0
                for i, vehicle in enumerate(vehicles):
                    vehicle.draw(screen)
                    vehicles_drawn += 1

                    # Optional: Draw small debug info (can be removed later)
                    if len(vehicles) < 10:  # Only show debug for few vehicles
                        debug_text = small_font.render(f"{vehicle.direction} #{i}", True, (255, 255, 0))
                        screen.blit(debug_text, (vehicle.rect.x, vehicle.rect.y - 15))

                # Draw UI
                ns_waiting, ew_waiting = controller.count_waiting_vehicles(vehicles)

                ui_lines = [
                    f"Episode: {episode + 1}/{episodes}",
                    f"Phase: {controller.current_phase} ({controller.phase_timer:.1f}s)",
                    f"Vehicles: {len(vehicles)} (Drawn: {vehicles_drawn}) (Total spawned: {stats['total_vehicles']})",
                    f"Waiting - NS: {ns_waiting}, EW: {ew_waiting}",
                    f"Mode: {mode.upper()}",
                    f"Epsilon: {controller.q_agent.eps:.3f}" if mode == "train" else "",
                    f"Spawn Timer: {spawn_timer:.1f}s"
                ]

                for i, line in enumerate(ui_lines):
                    if line:
                        text = font.render(line, True, (255, 255, 255))
                        screen.blit(text, (10, 10 + i * 25))

                pg.display.flip()

            frame_count += 1

        # End of episode
        stats['episode_rewards'].append(episode_reward)

        # Decay epsilon for training
        if mode == "train":
            controller.q_agent.decay_epsilon()

        # Save progress periodically
        if episode % 10 == 0:
            controller.q_agent.save()
            controller.save_data()
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    # Final save
    controller.q_agent.save()
    controller.save_data()

    print("Simulation complete!")
    return stats


# ---------------- CLI Interface ----------------
def main():
    parser = argparse.ArgumentParser(description="Smart Traffic Signal Simulation")
    parser.add_argument("--mode", choices=["train", "play"], default="play",
                        help="Training or playing mode")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to run")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without display (faster training)")
    parser.add_argument("--use-lstm", action="store_true",
                        help="Enable LSTM prediction")

    args = parser.parse_args()

    print("Starting Smart Traffic Signal Simulation")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"LSTM: {'Enabled' if args.use_lstm else 'Disabled'}")
    print(f"TensorFlow: {'Available' if TF_AVAILABLE else 'Not Available'}")

    try:
        stats = run_simulation(
            mode=args.mode,
            episodes=args.episodes,
            display=not args.no_display,
            use_lstm=args.use_lstm
        )

        print("\n=== Simulation Statistics ===")
        print(f"Total vehicles spawned: {stats['total_vehicles']}")
        if stats['episode_rewards']:
            avg_reward = sum(stats['episode_rewards']) / len(stats['episode_rewards'])
            print(f"Average episode reward: {avg_reward:.2f}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        pg.quit()


if __name__ == "__main__":
    main()