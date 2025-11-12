# Linear Regression (single feature) — Demo

This small demo shows simple linear regression (one feature) trained with batch gradient descent.

Files:

- `LAB_linear_regression.md` : explanation of hypothesis, cost function, and gradients.
- `scripts/linear_regression_demo.py` : runnable script that generates data, trains, and saves plots to `outputs/`.
- `requirements.txt` : Python dependencies.

How to run

1. (Optional) create a virtual environment, then install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo script (it will create `outputs/fit.png` and `outputs/cost.png`):

```bash
python3 scripts/linear_regression_demo.py
```

The saved images can be shown to students. Encourage them to change `lr` and `iters` in the script to explore behavior.
