"""
traffic_sim_integrated.py

- Modes:
    sim_train  : Train Q-learning on the Pygame multi-lane simulation and save q_table.npy
    sim_play   : Run the simulation and use a saved q_table (if present) to control the light
    real_run   : Run YOLO+SORT on a video and apply the saved q_table to choose light actions
    collect_counts : Run YOLO+SORT and save waiting counts to counts.csv for LSTM
    train_lstm : (optional) Train LSTM on counts.csv if TensorFlow is installed

Usage examples:
    python traffic_sim_integrated.py --mode sim_train --episodes 300 --display
    python traffic_sim_integrated.py --mode sim_play --display
    python traffic_sim_integrated.py --mode real_run --display --duration 60

Before running real_run or collect_counts, ensure ultralytics and sort.py are available.
"""

import argparse
import os
import pickle
import random
import sys
import time
from collections import deque

import numpy as np

# Optional libraries
try:
    import pygame
except Exception:
    pygame = None

try:
    import cv2
except Exception:
    cv2 = None

# For YOLO and SORT in real_run/collect_counts modes
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from sort import Sort
except Exception:
    Sort = None

# Optional LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -------------------- Config --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 700
FPS = 60

LANES = [
    {'x': 200, 'y_start': -100},
    {'x': 350, 'y_start': -200},
    {'x': 500, 'y_start': -150},
    {'x': 650, 'y_start': -250},
]  # multiple lanes (x positions). Cars move downward to the stop line

STOP_LINE_Y = 480   # where cars stop
CAR_W, CAR_H = 48, 28
CAR_SPEED = 3
SPAWN_INTERVAL_FRAMES = 90

YELLOW_DURATION_FRAMES = 40

# Discretization bins for waiting vehicles -> state index
STATE_BINS = [0, 1, 3, 6, 10, 999]
N_ACTIONS = 3  # 0=RED,1=YELLOW,2=GREEN

Q_TABLE_PATH = "q_table.npy"
COUNTS_CSV = "counts.csv"
LSTM_MODEL_PATH = "lstm_traffic.h5"

# For real video detection
VIDEO_PATH = "cars.mp4"          # change if needed
MODEL_PATH = "Yolo-Weights/yolov8n.pt"
MASK_PATH = None                 # optional mask path or None
FRAME_SKIP = 2                   # speedup for video processing

# -------------------- Utilities --------------------
def discretize_waiting(waiting):
    """Map waiting count to discrete state index using STATE_BINS"""
    for i in range(len(STATE_BINS) - 1):
        if STATE_BINS[i] <= waiting <= STATE_BINS[i + 1]:
            return i
    return len(STATE_BINS) - 2

# -------------------- Q Agent --------------------
class QAgent:
    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.95, eps=0.2):
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def act(self, state_idx):
        if np.random.rand() < self.eps:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s, a, r, s2):
        best_next = np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (r + self.gamma * best_next - self.Q[s, a])

    def save(self, path=Q_TABLE_PATH):
        np.save(path, self.Q)

    def load(self, path=Q_TABLE_PATH):
        self.Q = np.load(path)

# -------------------- Multi-lane Simulation --------------------
class TrafficSim:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, display=True, spawn_interval=SPAWN_INTERVAL_FRAMES):
        if pygame is None:
            raise RuntimeError("pygame is required for simulation. pip install pygame")
        import pygame as pg
        # initialize pygame only once
        if not pg.get_init():
            pg.init()
        if not pg.font.get_init():
            pg.font.init()

        self.pg = pg
        self.WIDTH, self.HEIGHT = width, height
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption("Multi-lane Traffic Simulation")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 28)
        self.display = display
        self.spawn_interval = spawn_interval
        self.reset()

    def reset(self):
        # car is dict: {'lane_idx':int, 'x':int, 'y':float, 'stopped':bool}
        self.cars = []
        self.frame = 0
        self.light_state = "RED"   # RED / YELLOW / GREEN
        self.yellow_timer = 0
        self.spawn_counter = 0

    def spawn_car(self):
        lane_idx = random.randrange(len(LANES))
        lane = LANES[lane_idx]
        x = lane['x']
        y = lane['y_start']
        car = {'lane_idx': lane_idx, 'x': x, 'y': y, 'stopped': False}
        self.cars.append(car)

    def step(self, action):
        """
        action: 0=RED, 1=YELLOW, 2=GREEN
        returns: state_idx, reward, waiting_count
        """
        # apply action
        if action == 1:  # YELLOW
            self.light_state = "YELLOW"
            self.yellow_timer = YELLOW_DURATION_FRAMES
        elif action == 2:
            self.light_state = "GREEN"
            self.yellow_timer = 0
        else:
            self.light_state = "RED"
            self.yellow_timer = 0

        # spawn logic
        self.spawn_counter += 1
        if self.spawn_counter >= self.spawn_interval:
            self.spawn_car()
            self.spawn_counter = 0

        # update cars motion / stopping
        waiting = 0
        newcars = []
        for car in self.cars:
            front_y = car['y'] + CAR_H
            # If RED or YELLOW (yellow acts like red for stopping), car stops at stop_line
            if self.light_state in ("RED", "YELLOW") and front_y >= STOP_LINE_Y - 5:
                car['stopped'] = True
                waiting += 1
            else:
                car['stopped'] = False
                car['y'] += CAR_SPEED
            # keep if not offscreen
            if car['y'] < self.HEIGHT + 100:
                newcars.append(car)
        self.cars = newcars

        # yellow timer -> after yellow ends, convert to RED (safety) unless next action changes
        if self.yellow_timer > 0:
            self.yellow_timer -= 1
            if self.yellow_timer == 0 and self.light_state == "YELLOW":
                self.light_state = "RED"

        reward = -waiting  # minimize waiting
        s_idx = discretize_waiting(waiting)
        return s_idx, reward, waiting

    def render(self):
        pg = self.pg
        self.screen.fill((30, 30, 40))
        # draw lanes
        for lane in LANES:
            x = lane['x']
            pg.draw.rect(self.screen, (50,50,50), pg.Rect(x-30, 0, 60, self.HEIGHT))
        # stop line
        pg.draw.line(self.screen, (255,255,255), (0, STOP_LINE_Y), (self.WIDTH, STOP_LINE_Y), 4)
        # traffic light
        box_x = self.WIDTH // 2 - 30
        box_y = STOP_LINE_Y - 160
        pg.draw.rect(self.screen, (20,20,20), pg.Rect(box_x, box_y, 60, 120))
        if self.light_state == "RED":
            col = (255, 0, 0)
        elif self.light_state == "YELLOW":
            col = (255, 255, 0)
        else:
            col = (0, 255, 0)
        pg.draw.circle(self.screen, col, (self.WIDTH//2, STOP_LINE_Y - 110), 18)

        # draw cars
        for car in self.cars:
            color = (0,120,255)
            rect = pg.Rect(int(car['x'] - CAR_W//2), int(car['y']), CAR_W, CAR_H)
            pg.draw.rect(self.screen, color, rect)
            if car['stopped']:
                pg.draw.rect(self.screen, (200,80,80), rect, 2)

        # info
        waiting = sum(1 for c in self.cars if c['stopped'])
        txt = self.font.render(f"Light: {self.light_state}  Waiting: {waiting}  Cars:{len(self.cars)}", True, (255,255,255))
        self.screen.blit(txt, (10, 10))

        if self.display:
            pg.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.pg:
            self.pg.quit()

# -------------------- Simulation training --------------------
def train_simulation_rl(episodes=300, steps_per_ep=800, display=True, save_path=Q_TABLE_PATH):
    sim = TrafficSim(display=display)
    n_states = len(STATE_BINS) - 1
    agent = QAgent(n_states=n_states, n_actions=N_ACTIONS, alpha=0.2, gamma=0.95, eps=0.3)

    for ep in range(episodes):
        sim.reset()
        total_reward = 0
        for t in range(steps_per_ep):
            # compute current state
            waiting = sum(1 for c in sim.cars if c['stopped'])
            s = discretize_waiting(waiting)
            a = agent.act(s)
            s2, r, waiting2 = sim.step(a)
            agent.update(s, a, r, s2)
            total_reward += r
            if display and (ep % 20 == 0):
                sim.render()
            # handle pygame events
            for event in sim.pg.event.get():
                if event.type == sim.pg.QUIT:
                    sim.close()
                    return agent
        # decay epsilon
        agent.eps = max(0.02, agent.eps * 0.995)
        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} total_reward={total_reward} eps={agent.eps:.3f}")
    # save Q-table
    agent.save(save_path)
    print("Training finished, Q-table saved to", save_path)
    sim.close()
    return agent

# -------------------- Simulation play with trained policy --------------------
def play_simulation(policy_path=Q_TABLE_PATH, display=True):
    sim = TrafficSim(display=display)
    agent = QAgent(n_states=len(STATE_BINS)-1, n_actions=N_ACTIONS)
    if os.path.exists(policy_path):
        agent.load(policy_path)
        print("Loaded policy from", policy_path)
    else:
        print("Policy not found, running with random actions")
    try:
        sim.reset()
        while True:
            waiting = sum(1 for c in sim.cars if c['stopped'])
            s = discretize_waiting(waiting)
            a = agent.act(s)
            sim.step(a)
            sim.render()
            for event in sim.pg.event.get():
                if event.type == sim.pg.QUIT:
                    sim.close()
                    return
    except KeyboardInterrupt:
        sim.close()

# -------------------- Video detection and apply policy --------------------
class VideoDetector:
    def __init__(self, video_path=VIDEO_PATH, model_path=MODEL_PATH, mask_path=MASK_PATH, stop_zone=None, frame_skip=FRAME_SKIP):
        if YOLO is None or Sort is None:
            raise RuntimeError("Ultralytics YOLO or SORT is not available. Install ultralytics and make sure sort.py is importable.")
        if cv2 is None:
            raise RuntimeError("OpenCV not available.")
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) if (mask_path and os.path.exists(mask_path)) else None
        # simple default stop zone if none provided: lane area near bottom center
        self.stop_zone = stop_zone if stop_zone else (200, STOP_LINE_Y - 80, 800, STOP_LINE_Y + 20)
        self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)
        self.frame_skip = frame_skip
        self.count_history = []

    def _prepare_mask(self, frame):
        if self.raw_mask is None:
            return None
        m = self.raw_mask.copy()
        if len(m.shape) == 3 and m.shape[2] == 4:
            m = cv2.cvtColor(m, cv2.COLOR_BGRA2BGR)
        if len(m.shape) == 3 and m.shape[2] == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if (m.shape[0], m.shape[1]) != (frame.shape[0], frame.shape[1]):
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
        _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
        return m.astype('uint8')

    def run(self, policy=None, display=True, duration_seconds=None):
        start = time.time()
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1
            if self.frame_skip > 1 and frame_idx % self.frame_skip != 0:
                continue

            mask = self._prepare_mask(frame)
            if mask is not None:
                img_region = cv2.bitwise_and(frame, frame, mask=mask)  # use mask param
            else:
                img_region = frame.copy()

            # YOLO inference (single image)
            results = self.model(img_region, stream=False)
            dets = np.empty((0, 5))
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0]) if len(box.cls) > 0 else -1
                    # use COCO indices for vehicles (car=2, motorcycle=3, bus=5, truck=7)
                    if cls in (2, 3, 5, 7) and conf > 0.25:
                        dets = np.vstack((dets, np.array([x1, y1, x2, y2, conf])))

            tracked = self.tracker.update(dets)

            # count waiting inside stop zone
            sx1, sy1, sx2, sy2 = self.stop_zone
            waiting_ids = set()
            for tr in tracked:
                x1, y1, x2, y2, tid = tr
                x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                if sx1 < cx < sx2 and sy1 < cy < sy2:
                    waiting_ids.add(tid)

            waiting_count = len(waiting_ids)
            self.count_history.append((time.time(), waiting_count))

            action_name = "N/A"
            if policy is not None:
                # policy could be numpy array Q-table or callable
                s_idx = discretize_waiting(waiting_count)
                if isinstance(policy, np.ndarray):
                    action = int(np.argmax(policy[s_idx]))
                elif callable(policy):
                    action = int(policy(s_idx))
                else:
                    action = int(np.argmax(policy[s_idx]))
                action_name = ("RED", "YELLOW", "GREEN")[action]
                # overlay chosen light color on frame
                color = (0, 0, 255) if action == 0 else (0, 255, 255) if action == 1 else (0, 255, 0)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 3)
            else:
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)

            cv2.putText(frame, f"Waiting: {waiting_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Action: {action_name}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2)

            if display:
                cv2.imshow("Real Detection", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            if duration_seconds and (time.time() - start) > duration_seconds:
                break

        # save counts
        if len(self.count_history) > 0:
            import pandas as pd
            df = pd.DataFrame(self.count_history, columns=['ts', 'waiting'])
            df.to_csv(COUNTS_CSV, index=False)
            print("Saved counts to", COUNTS_CSV)

        self.cap.release()
        cv2.destroyAllWindows()

# -------------------- LSTM helper (optional) --------------------
class LSTMPredictor:
    def __init__(self, lookback=10):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_sequences(self, counts):
        arr = np.array(counts).reshape(-1, 1).astype('float32')
        scaled = self.scaler.fit_transform(arr)
        X, y = [], []
        for i in range(len(scaled) - self.lookback):
            X.append(scaled[i:i + self.lookback, 0])
            y.append(scaled[i + self.lookback, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def build(self):
        m = Sequential()
        m.add(LSTM(32, input_shape=(self.lookback, 1)))
        m.add(Dense(16, activation='relu'))
        m.add(Dense(1))
        m.compile(optimizer='adam', loss='mse')
        self.model = m

    def train(self, counts, epochs=30):
        X, y = self.prepare_sequences(counts)
        if self.model is None:
            self.build()
        cb = [EarlyStopping(patience=5, restore_best_weights=True),
              ModelCheckpoint(LSTM_MODEL_PATH, save_best_only=True)]
        self.model.fit(X, y, epochs=epochs, batch_size=8, validation_split=0.1, callbacks=cb)
        print("LSTM trained, saved to", LSTM_MODEL_PATH)

# -------------------- CLI main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sim_train", "sim_play", "real_run", "collect_counts", "train_lstm"], required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--duration", type=int, default=None, help="duration in seconds for real_run")
    args = parser.parse_args()

    if args.mode == "sim_train":
        print("Training Q-table in simulation...")
        train_simulation_rl(episodes=args.episodes, steps_per_ep=args.steps, display=args.display)
        print("Done.")

    elif args.mode == "sim_play":
        play_simulation(policy_path=Q_TABLE_PATH, display=args.display)

    elif args.mode == "real_run":
        if YOLO is None or Sort is None:
            print("YOLO or SORT not available. Install ultralytics and have sort.py available.")
            return
        policy = None
        if os.path.exists(Q_TABLE_PATH):
            try:
                policy = np.load(Q_TABLE_PATH)
                print("Loaded Q-table from", Q_TABLE_PATH)
            except Exception as e:
                print("Failed to load Q-table:", e)
        detector = VideoDetector(video_path=VIDEO_PATH, model_path=MODEL_PATH, mask_path=MASK_PATH, stop_zone=(200, STOP_LINE_Y - 40, 800, STOP_LINE_Y + 40))
        detector.run(policy=policy, display=args.display, duration_seconds=args.duration)

    elif args.mode == "collect_counts":
        if YOLO is None or Sort is None:
            print("YOLO or SORT not available.")
            return
        detector = VideoDetector(video_path=VIDEO_PATH, model_path=MODEL_PATH, mask_path=MASK_PATH, stop_zone=(200, STOP_LINE_Y - 40, 800, STOP_LINE_Y + 40))
        detector.run(policy=None, display=args.display, duration_seconds=args.duration)

    elif args.mode == "train_lstm":
        if not TF_AVAILABLE:
            print("TensorFlow not available; install tensorflow to use LSTM.")
            return
        if not os.path.exists(COUNTS_CSV):
            print("counts.csv not found. Run collect_counts mode first.")
            return
        import pandas as pd
        df = pd.read_csv(COUNTS_CSV)
        if 'waiting' in df.columns:
            counts = df['waiting'].tolist()
        elif 'count' in df.columns:
            counts = df['count'].tolist()
        else:
            counts = df.iloc[:, 1].tolist()
        predictor = LSTMPredictor(lookback=10)
        predictor.train(counts, epochs=30)
    else:
        print("Unknown mode.")

if __name__ == "__main__":
    main()
