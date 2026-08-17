"""
Linear Regression From Scratch (Gradient Descent)
===================================================

This script teaches a computer to draw the "best fit" straight line through a
set of data points: hours studied (x) vs. test score (y).

The line has the form:
    y = m * x + b

Where:
    m = slope (how many extra points you get per extra hour studied)
    b = intercept (predicted score when hours studied = 0)

We don't know m and b at the start. We FIND them using an algorithm called
"gradient descent" - we start with a guess, and slowly improve it, over and
over, until the line fits the data well.

This script is meant to be read top to bottom like a story. Every step is
explained in comments right above the code that does it.
"""

# -------------------------------------------------------------------------
# STEP 0: Import the libraries we need
# -------------------------------------------------------------------------
# numpy   -> lets us do fast math on lists of numbers (called "arrays")
# pandas  -> lets us load and work with our CSV file as a table
# matplotlib.pyplot -> lets us draw graphs / plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# STEP 1: Load the dataset
# -------------------------------------------------------------------------
# pd.read_csv() reads a CSV file and turns it into a table (called a
# DataFrame). Our CSV has two columns: hours_studied and test_score.
data = pd.read_csv("study_scores.csv")

# Let's print the first few rows just to see what the data looks like.
# .head() shows the first 5 rows by default.
print("First 5 rows of our dataset:")
print(data.head())
print()  # print an empty line for spacing

# Pull the two columns out as separate numpy arrays, since that's easier
# to do math with than a pandas table.
# .values converts a pandas column into a plain numpy array of numbers.
x = data["hours_studied"].values   # our INPUT: hours studied
y = data["test_score"].values      # our OUTPUT: actual test score


# -------------------------------------------------------------------------
# STEP 2: Look at the data before doing anything else
# -------------------------------------------------------------------------
# It's always a good idea to SEE your data before modeling it.
# plt.scatter() draws a dot for every (x, y) pair in our data.
plt.figure(figsize=(8, 5))          # set the size of the plot window
plt.scatter(x, y, color="steelblue", alpha=0.7)
plt.xlabel("Hours Studied")         # label for the x-axis
plt.ylabel("Test Score")            # label for the y-axis
plt.title("Raw Data: Hours Studied vs. Test Score")
# add a light grid to make it easier to read
plt.grid(True, alpha=0.3)
plt.savefig("plot_1_raw_data.png")  # save the plot as an image file
print("Saved plot_1_raw_data.png - open this to see your raw data.")
print()


# -------------------------------------------------------------------------
# STEP 3: Define the "cost function" (how wrong is our line?)
# -------------------------------------------------------------------------
# This function tells us how BAD our current line (m, b) is at predicting
# the real scores. Lower cost = better line.
#
# How it works, for EVERY data point:
#   1. Predict a score using our current line:      prediction = m*x + b
#   2. Find the error:                                error = actual - prediction
#   3. Square the error (so negatives don't cancel):  error^2
# Then we AVERAGE all those squared errors. This is called
# "Mean Squared Error" (MSE), and it's the industry-standard way to measure
# how good a regression line is.
def compute_cost(m, b, x, y):
    # m * x does the multiplication for EVERY value in x at once (numpy magic)
    predictions = m * x + b

    # subtract element-by-element: actual score minus predicted score
    errors = y - predictions

    # square every error, then take the average of all of them
    mse = np.mean(errors ** 2)

    return mse


# -------------------------------------------------------------------------
# STEP 4: Define one "step" of gradient descent
# -------------------------------------------------------------------------
# This function looks at the current m and b, figures out which direction
# would REDUCE the cost (make the line fit better), and moves m and b a
# tiny bit in that direction.
#
# The "gradient" is just calculus telling us the slope of the cost function
# itself (not to be confused with the slope of our regression line!).
# We don't need to derive this by hand - the formulas below are the
# standard gradient descent update rules for linear regression.
def gradient_step(m, b, x, y, learning_rate):
    n = len(x)  # how many data points we have

    # Current predictions using our current m and b
    predictions = m * x + b

    # Current errors (how far off each prediction is)
    errors = y - predictions

    # Gradient (direction of steepest INCREASE in cost) for m and b.
    # We use the negative of this to DECREASE the cost instead.
    dm = (-2 / n) * np.sum(x * errors)   # gradient with respect to m
    db = (-2 / n) * np.sum(errors)       # gradient with respect to b

    # Update m and b by taking a small step opposite to the gradient.
    # "learning_rate" controls how big each step is.
    m_new = m - learning_rate * dm
    b_new = b - learning_rate * db

    return m_new, b_new


# -------------------------------------------------------------------------
# STEP 5: Train the model (run gradient descent many times)
# -------------------------------------------------------------------------
# We start with a bad guess (a flat line at 0) and let gradient descent
# improve it, one small step at a time.
m = 0.0   # starting slope guess
b = 0.0   # starting intercept guess

learning_rate = 0.01   # how big each update step is (small = safe but slow)
n_iterations = 1000    # how many times we update m and b

# We'll keep a record of the cost at every iteration, so we can plot it
# later and SEE the model improving over time.
cost_history = []

print("Training the model...")
for i in range(n_iterations):
    # Take one gradient descent step, updating m and b
    m, b = gradient_step(m, b, x, y, learning_rate)

    # Measure how good the line is now, and save it
    current_cost = compute_cost(m, b, x, y)
    cost_history.append(current_cost)

    # Every 100 iterations, print progress so we can watch it improve
    if i % 100 == 0:
        print(
            f"  Iteration {i:4d}:  m = {m:.3f}  b = {b:.3f}  cost = {current_cost:.3f}")

print()
print(f"Training finished!")
print(f"Final line:  test_score = {m:.3f} * hours_studied + {b:.3f}")
print()


# -------------------------------------------------------------------------
# STEP 6: Plot the cost decreasing over time
# -------------------------------------------------------------------------
# This proves the model actually LEARNED - the cost should start high and
# drop quickly, then flatten out near the end.
plt.figure(figsize=(8, 5))
plt.plot(cost_history, color="firebrick")
plt.xlabel("Iteration")
plt.ylabel("Cost (Mean Squared Error)")
plt.title("Cost Decreasing as the Model Trains")
plt.grid(True, alpha=0.3)
plt.savefig("plot_2_cost_over_time.png")
print("Saved plot_2_cost_over_time.png - the cost should drop and flatten out.")
print()


# -------------------------------------------------------------------------
# STEP 7: Plot our fitted line over the original data
# -------------------------------------------------------------------------
# This is the "money shot" - does our learned line actually match the data?
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="steelblue", alpha=0.7, label="Actual data")

# Create 100 evenly-spaced x-values across the range of our data, so we can
# draw a smooth line (not just connect our original dots).
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b   # use our TRAINED m and b to predict along the line

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


# -------------------------------------------------------------------------
# STEP 8: Use the trained model to make new predictions
# -------------------------------------------------------------------------
# Now that we have a trained m and b, we can predict scores for NEW
# students who aren't even in our dataset.
# let's predict for students who studied 2, 5, and 8 hours
new_hours = [2, 5, 8]

print("Predictions for new students:")
for hours in new_hours:
    predicted_score = m * hours + b
    print(
        f"  A student who studied {hours} hours is predicted to score {predicted_score:.1f}")

print()
print("Done! Check the 3 saved PNG images in this folder to see the plots.")
