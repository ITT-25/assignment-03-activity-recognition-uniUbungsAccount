import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from DIPPID import SensorUDP

PORT             = 5700
NAME             = "david"
ACTIONS          = ["running", "rowing", "jumpingjack", "lifting"]
REPS             = 5
SECONDS_PER_REP  = 10
RATE_HZ          = 100
SAMPLES_PER_REP  = SECONDS_PER_REP * RATE_HZ

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
IMG_DIR    = os.path.join(SCRIPT_DIR, "img")

HEADERS = [
    "id", "timestamp",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]

IMAGES = {
    "running":     "running_1.png",
    "rowing":      "rowing_1.png",
    "jumpingjack": "jumpingjack_1.png",
    "lifting":     "lifting_1.png"
}

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def show_image(action):
    fname = IMAGES.get(action)
    full = os.path.join(IMG_DIR, fname) if fname else None

    plt.ion()
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(mpimg.imread(full))
    ax.axis("off")
    ax.set_title(f"{action}", pad=10)
    fig.canvas.manager.set_window_title(action)
    plt.show()
    plt.pause(0.1)
    return fig, ax

def update_message(ax, msg):
    for t in ax.texts:
        t.remove()
    ax.text(
        0.5, -0.1, msg,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=12, color="red"
    )
    ax.figure.canvas.draw()
    plt.pause(0.01)

def wait_for_tap(sensor):
    while not sensor.get_value("button_1"):
        time.sleep(0.05)
    while sensor.get_value("button_1"):
        time.sleep(0.05)

make_dir(DATA_DIR)
sensor = SensorUDP(PORT)

for action in ACTIONS:
    fig, ax = show_image(action)

    for rep in range(1, REPS + 1):
        fname = f"{NAME}-{action}-{rep}.csv"
        path  = os.path.join(DATA_DIR, fname)
        pd.DataFrame(columns=HEADERS).to_csv(path, index=False)

        update_message(ax, f"Rep {rep}/{REPS} — tap button_1 to start")
        wait_for_tap(sensor)
        update_message(ax, "Recording…")

        t0 = time.perf_counter()
        for i in range(SAMPLES_PER_REP):
            acc  = sensor.get_value("accelerometer")
            gyro = sensor.get_value("gyroscope")

            pd.DataFrame([{
                "id":        i,
                "timestamp": time.time(),
                "acc_x":     acc["x"],
                "acc_y":     acc["y"],
                "acc_z":     acc["z"],
                "gyro_x":    gyro["x"],
                "gyro_y":    gyro["y"],
                "gyro_z":    gyro["z"]
            }]).to_csv(path, mode="a", header=False, index=False)

            if i % 10 == 0:
                fig.canvas.flush_events()

            next_t = t0 + (i+1)/RATE_HZ
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

    if fig:
        plt.close(fig)

print("Data gathered, hope it was nice workout")
