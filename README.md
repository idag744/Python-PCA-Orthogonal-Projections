# Orthogonal Projections in Python

This project demonstrates the use of **orthogonal projections** in linear algebra and their applications in:
- Vector projections in 1D and higher-dimensional subspaces
- Face reconstruction using **Eigenfaces** (PCA basis)
- Least Squares Regression as a projection problem

It is structured as a small but professional Python package, suitable for showcasing intermediate-level coding skills to recruiters.

---

## Features
- Projection functions (`projection_matrix_1d`, `project_1d`, `projection_matrix_general`, `project_general`)
- Test suite with invariants and correctness checks
- Demonstrations:
  - Eigenfaces reconstruction on the Olivetti Faces dataset
  - Linear regression as an application of projections

---

## Installation
```bash
git clone https://github.com/yourusername/orthogonal-projections.git
cd orthogonal-projections
pip install -r requirements.txt

## Structure
orthogonal-projections/
│── projections/
│   ├── __init__.py
│   ├── core.py              # projection functions
│   ├── utils.py             # test utilities
│── demos/
│   ├── eigenfaces_demo.py   # eigenfaces reconstruction
│   ├── regression_demo.py   # least squares regression
│── tests/
│   ├── test_projections.py
│── eigenfaces.npy           # eigenface basis (kept local)
│── requirements.txt
│── README.md
│── .gitignore


## Run tests
pytest tests/

## Run demos
python demos/eigenfaces_demo.py
python demos/regression_demo.py
