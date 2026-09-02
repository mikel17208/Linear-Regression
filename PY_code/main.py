import os  # use to interact the file path and folders
import numpy as np  # use for fast math on number arrays
import pandas as pd  # use for loading with tabular data like csv and spreadsheet file
import matplotlib.pyplot as plt  # make graphs and charts

# make it so the data can be read from other folder and show where the path is
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "Data", "study_scores.csv")
data = pd.read_csv(csv_path)  # load the csv into a table

# checking the data, shows only first 5 rows by default
print(data.head())

# this pulls each column into its own numpy array
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
print()  # blank line for spacing

# measures progress, not doing the learning itself


def compute_cost(m, b, x, y):
    # predict a score then find how far off each prediction
    predictions = m * x + b
    errors = y - predictions
    # average of the errors squared
    mse = np.mean(errors ** 2)
    return mse  # send the cost value back out

# learning happens here
# current m and b, then the data and learning rate


def gradient_step(m, b, x, y, learning_rate):
    # counts how many student in the data
    n = len(x)
    # same as compute_cost
    predictions = m * x + b
    errors = y - predictions
    # -2 is constant, np.sum is to add all arrays
    dm = (-2 / n) * np.sum(x * errors)  # how much to adjust the slope (m)
    db = (-2 / n) * np.sum(errors)  # how much to adjust the intercept (b)
    m_new = m - learning_rate * dm  # apply the adjustment to m
    b_new = b - learning_rate * db  # apply the adjustment to b
    return m_new, b_new  # send back the improved m and b


# starting guess for m and b, a flat line at zero
m = 0.0
b = 0.0

# how big each update step is
learning_rate = 0.01
# how many times i run the gradient_step
n_iterations = 1000

# collect the cost value from every single iteration
cost_history = []

# shows message that it is training
print("Training the model...")
# start a loop 0 to 999
for i in range(n_iterations):
    # calls the gradient step and overwrites the m and b with improved values
    m, b = gradient_step(m, b, x, y, learning_rate)

    # it's purely for monitoring/measuring progress
    current_cost = compute_cost(m, b, x, y)
    # saves that cost value into the list we created earlier
    cost_history.append(current_cost)

    # is the remainder of i divided by 100 equal to zero (true every 100th loop)
    if i % 100 == 0:
        # shows the current progress of training
        print(
            f"  Iteration {i:4d}:  m = {m:.3f}  b = {b:.3f}  cost = {current_cost:.3f}")

print()
print(f"Training finished!")
print(f"Final line:  test_score = {m:.3f} * hours_studied + {b:.3f}")
print()

# sets the size
plt.figure(figsize=(8, 5))
# line plot of the cost at every iteration, in red
plt.plot(cost_history, color="firebrick")
# labels
plt.xlabel("Iteration")
plt.ylabel("Cost (Mean Squared Error)")
# title
plt.title("Cost Decreasing as the Model Trains")
# turn on the grid line and the color to faint black lines
plt.grid(True, alpha=0.3)
# saved as an image called this
plt.savefig("plot_2_cost_over_time.png")
# confirming that the file was saved
print("Saved plot_2_cost_over_time.png - the cost should drop and flatten out.")
print()

# new blank plot, then scatter the real data points again
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="steelblue", alpha=0.7, label="Actual data")

# np.linspace(start, stop, count)
# generates 100 evenly-spaced numbers, so the line draws smooth not jagged
x_line = np.linspace(x.min(), x.max(), 100)
# use the final trained m and b to compute the line's y-values
y_line = m * x_line + b

# draw the fitted line in red on top of the scatter plot
plt.plot(x_line, y_line, color="firebrick",
         linewidth=2, label="Our fitted line")
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.title("Fitted Line vs. Actual Data")
plt.legend()  # show the box naming which color is which
plt.grid(True, alpha=0.3)
plt.savefig("plot_3_fitted_line.png")
print("Saved plot_3_fitted_line.png - our line should cut through the middle of the dots.")
print()

# hypothetical new students not in the original dataset
new_hours = [2, 5, 8]

print("Predictions for new students:")
for hours in new_hours:
    # plug each new hours value into the trained line to predict a score
    predicted_score = m * hours + b
    print(
        f"  A student who studied {hours} hours is predicted to score {predicted_score:.1f}")

print()
print("Done! Check the 3 saved PNG images in this folder to see the plots.")