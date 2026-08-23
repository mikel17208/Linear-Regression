# lets python to interact with os to file path and folder
import os
# fast math on number arrays as np for short
import numpy as np
# for loading and working with tabular data like spreadsheet and csv file
# as pd
import pandas as pd
# for making graphs and charts as plt
import matplotlib.pyplot as plt

# folder where main.py itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# full path to the CSV file, built from that folder
csv_path = os.path.join(script_dir, "..", "Data", "study_scores.csv")
# load the CSV into a pandas table
data = pd.read_csv(csv_path)

print("First 5 rows of our dataset:")
print(data.head())
# spacing
print()

x = data["hours_studied"].values
y = data["test_score"].values

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="steelblue", alpha=0.7)
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.title("Raw Data: Hours Studied vs. Test Score")
plt.grid(True, alpha=0.3)
plt.savefig("plot_1_raw_data.png")
print("Saved plot_1_raw_data.png - open this to see your raw data.")
print()


def compute_cost(m, b, x, y):
    predictions = m * x + b
    errors = y - predictions
    mse = np.mean(errors ** 2)
    return mse


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

cost_history = []

print("Training the model...")
for i in range(n_iterations):
    m, b = gradient_step(m, b, x, y, learning_rate)

    current_cost = compute_cost(m, b, x, y)
    cost_history.append(current_cost)

    if i % 100 == 0:
        print(
            f"  Iteration {i:4d}:  m = {m:.3f}  b = {b:.3f}  cost = {current_cost:.3f}")

print()
print(f"Training finished!")
print(f"Final line:  test_score = {m:.3f} * hours_studied + {b:.3f}")
print()

plt.figure(figsize=(8, 5))
plt.plot(cost_history, color="firebrick")
plt.xlabel("Iteration")
plt.ylabel("Cost (Mean Squared Error)")
plt.title("Cost Decreasing as the Model Trains")
plt.grid(True, alpha=0.3)
plt.savefig("plot_2_cost_over_time.png")
print("Saved plot_2_cost_over_time.png - the cost should drop and flatten out.")
print()

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="steelblue", alpha=0.7, label="Actual data")

x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b

plt.plot(x_line, y_line, color="firebrick",
         linewidth=2, label="Our fitted line")
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.title("Fitted Line vs. Actual Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_3_fitted_line.png")
print("Saved plot_3_fitted_line.png - our line should cut through the middle of the dots.")
print()

new_hours = [2, 5, 8]

print("Predictions for new students:")
for hours in new_hours:
    predicted_score = m * hours + b
    print(
        f"  A student who studied {hours} hours is predicted to score {predicted_score:.1f}")

print()
print("Done! Check the 3 saved PNG images in this folder to see the plots.")
