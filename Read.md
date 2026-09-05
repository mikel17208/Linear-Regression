Linear Regression from Scratch
A simple, dependency-light script that fits a straight line to hours_studied vs. test_score data using gradient descent implemented from scratch, no scikit-learn, no black boxes. It's meant as a learning tool for understanding how linear regression actually works under the hood.

## Table of Contents
- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Example output](#example-output)
- [Customizing](#customizing)

## What it does
Loads study data from ../Data/study_scores.csv
Fits a line test_score = m * hours_studied + b by minimizing mean squared error (MSE) via gradient descent
Trains for 1000 iterations, logging progress every 100 steps
Saves three plots to visualize the process and result
Uses the trained model to predict scores for a few example students

## How it works
The goal is to find the straight line (test_score = m * hours_studied + b) that best fits the data. "Best" means the line that's closest, on average, to all the actual data points. There are two moving parts: m (the slope, how much score increases per hour studied) and b (the intercept, the predicted score at zero hours studied).

Here's the intuition, step by step:

1. Start with a guess. The script starts with a flat, useless line: m = 0, b = 0. This predicts a score of 0 no matter how many hours someone studied, obviously wrong, but it has to start somewhere.

2. Measure how wrong the guess is. This is what compute_cost() does. For every data point, it compares the line's prediction to the actual test score, and measures the size of that gap. It squares each gap (so negative and positive errors don't cancel out, and bigger mistakes are punished more) and averages them. This single number is the "cost", think of it as a score for how bad the current line is. Lower is better.

3. Figure out which way to nudge the line. This is what gradient_step() does. For the current m and b, it works out: if I increase m slightly, does the cost go up or down? Same question for b. This direction-and-size information is called the gradient. The line is then adjusted a small step in whichever direction reduces the cost, this step size is controlled by learning_rate.

4. Repeat, a lot. Steps 2 and 3 happen over and over (1000 times by default). Each round, the line shifts a little closer to fitting the data well. Early on, the cost usually drops fast because the starting guess is so far off. Later, the improvements get smaller as the line settles near its best fit, this is why plot_2_cost_over_time.png typically looks like a steep drop that flattens into a plateau.

5. Use the final line to make predictions. Once training is done, m and b represent the best-fit line found. Plugging in a new hours_studied value gives a predicted test_score, this is what happens in the final loop over new_hours.

This whole process, guess, measure error, adjust, repeat, is called gradient descent, and it's one of the core techniques behind how most machine learning models are trained, not just simple linear regression.

Glossary
Linear regression, a method for modeling the relationship between two variables as a straight line, so you can predict one from the other.
Gradient descent, an iterative optimization technique that repeatedly nudges a model's parameters in the direction that reduces error, a small step at a time, until it settles on a good fit.
Cost / cost function, a single number representing how wrong a model's current predictions are. Lower cost means a better fit. Also called a "loss function."
Mean Squared Error (MSE), the specific cost function used here. It squares each prediction's error (so all errors count as positive, and big mistakes count extra) and averages them across all data points.
Slope (m), how steeply the line rises; here, how many extra test-score points you get per additional hour studied.
Intercept (b), where the line crosses the y-axis; here, the predicted score for a student who studied zero hours.
Gradient, the direction and rate at which the cost changes as you adjust a parameter. Gradient descent uses this to know which way to move m and b to reduce the cost.
Learning rate, a tuning knob controlling how big each adjustment step is. Too large and training can overshoot and become unstable; too small and training takes a long time to converge.
Iteration, one full round of measuring the cost and adjusting m and b. This script runs 1000 of them.
Convergence, the point where further training barely changes the cost anymore, meaning the model has settled on (close to) its best fit.

## Project structure
This script expects the following layout:

your-repo/
├── Data/
│   └── study_scores.csv
└── Scripts/
    └── your_script_name.py   ← this script
The CSV needs at least two columns: hours_studied and test_score.

## Requirements
Python 3.7+
Dependencies:
pip install numpy pandas matplotlib

## Usage
From inside the script's folder:

python your_script_name.py
The script will print the first few rows of the loaded data, then train the model, printing progress like:

Iteration    0:  m = 0.412  b = 0.089  cost = 245.318
Iteration  100:  m = 3.021  b = 1.774  cost = 42.105
...

## Example output
Three PNG plots are saved to the folder you run the script from:

| File | Shows |
|---|---|
| plot_1_raw_data.png | Raw scatter plot of hours studied vs. test score, before any fitting |
| plot_2_cost_over_time.png | Cost (MSE) decreasing across training iterations, should drop fast then flatten |
| plot_3_fitted_line.png | Final fitted line drawn over the actual data points |
The console also prints the final line equation and sample predictions, e.g.:

Final line:  test_score = 4.812 * hours_studied + 50.203

Predictions for new students:
  A student who studied 2 hours is predicted to score 59.6
  A student who studied 5 hours is predicted to score 74.3
  A student who studied 8 hours is predicted to score 88.7

## Customizing
A few easy things to tweak at the top of the training loop:

learning_rate, how big each gradient step is. Too high can cause the cost to diverge instead of decrease; too low makes training slow.
n_iterations, how many training steps to run. Watch plot_2_cost_over_time.png to see if it's flattened out (converged) or still decreasing.
new_hours, the list of example hour values to predict scores for at the end.
