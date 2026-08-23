import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "Data", "study_scores.csv")
data = pd.read_csv(csv_path)
x = data["hours_studied"].values
y = data["test_score"].values


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


m = 0.0
b = 0.0
learning_rate = 0.01
n_iterations = 1000

history = []

print("Training the model and recording every step...")
for i in range(n_iterations):
    m, b = gradient_step(m, b, x, y, learning_rate)
    cost = compute_cost(m, b, x, y)
    history.append((m, b, cost))

print(f"Training finished! Final line: y = {m:.3f}x + {b:.3f}")
print()

frame_step = 10
frames_to_show = history[::frame_step]
print(f"Building an animation with {len(frames_to_show)} frames...")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(x, y, color="steelblue", alpha=0.7, label="Actual data")

x_line = np.linspace(x.min(), x.max(), 100)

line, = ax.plot([], [], color="firebrick", linewidth=2,
                label="Model's current line")

info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top",
                    fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Test Score")
ax.set_title("Watching Gradient Descent Learn")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)


def update(frame_index):
    m_frame, b_frame, cost_frame = frames_to_show[frame_index]

    y_line = m_frame * x_line + b_frame
    line.set_data(x_line, y_line)

    iteration_number = frame_index * frame_step
    info_text.set_text(
        f"Iteration: {iteration_number}\n"
        f"m (slope):     {m_frame:.3f}\n"
        f"b (intercept): {b_frame:.3f}\n"
        f"cost (MSE):    {cost_frame:.2f}"
    )

    return line, info_text


ani = animation.FuncAnimation(
    fig, update, frames=len(frames_to_show), interval=50, blit=True
)

output_filename = "gradient_descent_animation.gif"
ani.save(output_filename, writer="pillow", fps=20)

print(f"Saved {output_filename} - open this file and watch the line learn!")
