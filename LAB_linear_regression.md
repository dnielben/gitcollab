## Linear Regression (single feature) — Lab

This short lab introduces linear regression with one feature (also called simple linear regression). We'll cover the dataset, the hypothesis, the cost function, and gradient descent (including derivatives). A runnable Python demo accompanies this lab and produces images you can show to students.

### 1) Dataset

- We use a dataset of house sizes (single feature x) and house prices (target y).
- Example values (x measured in 1000s of square feet; y in thousands of dollars):

| x (1000 sqft) | y (price in $1000s) |
|---:|---:|
| 0.5 | 120 |
| 0.75 | 150 |
| 1.0 | 200 |
| 1.25 | 230 |
| 1.5 | 255 |

In the demo we generate a synthetic dataset with noise to illustrate training.

### 2) Hypothesis (model)

We model the relation between x and y with a straight line (affine function) parameterized by weight w and bias b:

$$
f_{w,b}(x) = w x + b
$$

- w is the slope (how much y changes per unit x).
- b is the intercept (value of y when x = 0).

Prediction for an input x is y_hat = f_{w,b}(x).

### 3) Cost function (Mean Squared Error)

To measure how well parameters (w,b) fit the dataset of m examples {(x^{(i)}, y^{(i)})}, we use the mean squared error (MSE). For gradient derivation we use the commonly-seen scaled form:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^m \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2
$$

- The factor 1/2 simplifies the gradients (cancels the 2 when differentiating).
- Goal: find (w,b) that minimize J(w,b).

### 4) Gradient Descent (batch)

Gradient descent iteratively updates w and b in the direction of steepest descent of J. With learning rate (step size) \(\alpha\), the updates are:

$$
w := w - \alpha \frac{\partial J}{\partial w},\qquad b := b - \alpha \frac{\partial J}{\partial b}
$$

Compute derivatives (vectorized / summed form):

Let predictions \(\hat y^{(i)} = f_{w,b}(x^{(i)}) = w x^{(i)} + b\).

The partial derivatives are:

$$
\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^m (\hat y^{(i)} - y^{(i)}) x^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat y^{(i)} - y^{(i)})
$$

So a full batch gradient descent update becomes:

$$
w := w - \alpha \left(\frac{1}{m} \sum_{i=1}^m (\hat y^{(i)} - y^{(i)}) x^{(i)}\right)
$$

$$
b := b - \alpha \left(\frac{1}{m} \sum_{i=1}^m (\hat y^{(i)} - y^{(i)})\right)
$$

### 5) Intuition and tips

- If learning rate \(\alpha\) is too large, gradient descent may diverge; if too small, convergence is slow.
- Feature scaling can help for multiple features; here, with one feature, scaling is optional but useful when units are large.
- Plotting the cost vs iterations helps check convergence.

### 6) Files in this lab

- `scripts/linear_regression_demo.py` : runnable Python demo that generates a synthetic house prices dataset, fits a line using batch gradient descent, and saves images (scatter+fit, cost vs iterations).
- `requirements.txt` : Python packages required.
- `README.md` : brief run instructions.

### 7) Learning checkpoints (what students should observe)

1. After a few iterations, the fitted line should approach the underlying true line (when noise is moderate).
2. Cost decreases monotonically when learning rate is well-chosen.
3. Changing learning rate shows underfitting/overshoot behavior.

---

Use the demo script to generate visual outputs you can show in class. See the README for run steps.
