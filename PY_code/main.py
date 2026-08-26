import os  # use to interact the file path and folders
import numpy as np  # use for fast math on number arrays
import pandas as pd  # use for loading with tabular data like csv and spreadsheet file
import matplotlib.pyplot as plt  # make graphs and charts

# make it so the data can be read from other folder and show where the path is
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "Data", "study_scores.csv")
data = pd.read_csv(csv_path)

# checking for the head only first 10 rows
print(data.head())

# this label your data into column and into padas arrays
x = data["hours_studied"].values
y = data["test_score"].values

# sets the size
plt.figure(figsize=(8, 5))
# scatter the data in blue color
plt.scatter(x, y, color="steelblue", alpha=0.7)
# label the 2 data
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
# title of the chart
plt.title("Raw Data: Hours Studied vs. Test Score")
# turn on the grid line and the color to faint black lines
plt.grid(True, alpha=0.3)
# saved as an image called this
plt.savefig("plot_1_raw_data.png")
# confirming that the file was saved
print("Saved plot_1_raw_data.png - open this to see your raw data.")
print()

# this function predictions to the real scores,
# and returns one number (MSE) showing how wrong
# that guess was on average
# study this logic


def compute_cost(m, b, x, y):
    predictions = m * x + b
    errors = y - predictions
    mse = np.mean(errors ** 2)
    return mse

# study this logic


def gradient_step(m, b, x, y, learning_rate):
    n = len(x)
    predictions = m * x + b
    errors = y - predictions
    dm = (-2 / n) * np.sum(x * errors)
    db = (-2 / n) * np.sum(errors)
    m_new = m - learning_rate * dm
    b_new = b - learning_rate * db
    return m_new, b_new
