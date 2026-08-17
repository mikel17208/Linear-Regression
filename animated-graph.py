"""
Linear Regression From Scratch - ANIMATED VERSION
====================================================

This is the same gradient descent training as linear_regression.py, but
instead of just showing the BEFORE and AFTER plots, this version creates an
ANIMATED GIF showing the line move and adjust in real time as it trains -
just like what you saw in the video.

Watching the line rotate/shift frame by frame is the best way to build
intuition for what gradient descent is actually doing: it starts with a bad
guess, and gradually rotates/shifts the line until it fits the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# -------------------------------------------------------------------------
# STEP 1: Load the dataset (same as before)
# -------------------------------------------------------------------------
data = pd.read_csv("study_scores.csv")
x = data["hours_studied"].values
y = data["test_score"].values


# -------------------------------------------------------------------------
# STEP 2: Same cost function and gradient step as before
# -------------------------------------------------------------------------
def compute_cost(m, b, x, y):
    predictions = m * x + b
    errors = y - predictions
    return np.mean(errors ** 2)


def gradient_step(m, b, x, y, learning_rate):
    n = len(x)
    predictions = m * x + b
    errors = y - predictions
    dm = (-2 / n) * np.sum(x * errors)
    db = (-2 / n) * np.sum(errors)
    m_new = m - learning_rate * dm
    b_new = b - learning_rate * db
    return m_new, b_new


# -------------------------------------------------------------------------
# STEP 3: Train the model, but this time SAVE a snapshot of (m, b) at
# every iteration so we can play them back as an animation afterward.
# -------------------------------------------------------------------------
m = 0.0
b = 0.0
learning_rate = 0.01
n_iterations = 1000

# history will store a snapshot after EVERY iteration: (m, b, cost)
history = []

print("Training the model and recording every step...")
for i in range(n_iterations):
    m, b = gradient_step(m, b, x, y, learning_rate)
    cost = compute_cost(m, b, x, y)
    history.append((m, b, cost))

print(f"Training finished! Final line: y = {m:.3f}x + {b:.3f}")
print()

# We have 1000 snapshots, but a 1000-frame GIF would be huge and slow.
# Instead, we'll only show every 10th snapshot (100 frames total) - still
# looks smooth, but the file size stays reasonable.
frame_step = 10
frames_to_show = history[::frame_step]
print(f"Building an animation with {len(frames_to_show)} frames...")


# -------------------------------------------------------------------------
# STEP 4: Build the animation
# -------------------------------------------------------------------------
# Set up the figure once, before the animation starts.
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the actual data points (these never change/move)
ax.scatter(x, y, color="steelblue", alpha=0.7, label="Actual data")

# x-values to draw our line across (the full width of the graph)
x_line = np.linspace(x.min(), x.max(), 100)

# Create ONE line object that we will update every frame, instead of
# redrawing a brand new line each time (this is faster and standard
# practice for matplotlib animations).
line, = ax.plot([], [], color="firebrick", linewidth=2,
                label="Model's current line")

# A text label in the corner to show the current m, b, and cost values,
# also updated every frame.
info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top",
                    fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Test Score")
ax.set_title("Watching Gradient Descent Learn")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)


def update(frame_index):
    """
    This function runs once per frame of the animation.
    matplotlib calls it automatically, passing in the frame number.
    """
    m_frame, b_frame, cost_frame = frames_to_show[frame_index]

    # Recompute the line's y-values using this frame's m and b
    y_line = m_frame * x_line + b_frame
    line.set_data(x_line, y_line)

    # Update the info text box
    iteration_number = frame_index * frame_step
    info_text.set_text(
        f"Iteration: {iteration_number}\n"
        f"m (slope):     {m_frame:.3f}\n"
        f"b (intercept): {b_frame:.3f}\n"
        f"cost (MSE):    {cost_frame:.2f}"
    )

    # We must return the objects that changed, for matplotlib's animation
    # engine to know what to redraw.
    return line, info_text


# Build the actual animation object.
# - fig: which figure to animate
# - update: the function to call for each frame
# - frames: how many frames total
# - interval: milliseconds between frames (50ms = fairly fast paced)
# - blit=True: a performance optimization (only redraw what changed)
ani = animation.FuncAnimation(
    fig, update, frames=len(frames_to_show), interval=50, blit=True
)

# Save the animation as a GIF file using the "pillow" writer.
# fps=20 means 20 frames per second when played back.
output_filename = "gradient_descent_animation.gif"
ani.save(output_filename, writer="pillow", fps=20)

print(f"Saved {output_filename} - open this file and watch the line learn!")
