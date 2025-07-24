# CURSORRULES.md

## Coding Standards & Conventions

### Language & Version
- Use **Python 3.8+** (recommended: 3.10+).
- All code must be compatible with the Python version specified in `.python-version`.

### Code Style
- Follow **PEP8** for all Python code.
- Use **black** for automatic code formatting.
- Use **flake8** for linting and code quality checks.
- Use **isort** for import sorting.
- Use **autopep8** for additional formatting if needed.
- Type annotations are required; check with **mypy**.
- Use descriptive, snake_case for variables and functions; PascalCase for classes.

### Project Structure
- Organize code into logical modules under `src/`:
  - `data.py`: Data loading and preprocessing
  - `regime.py`: Regime detection (HMM, LSTM, ensemble)
  - `deep_learning.py`: LSTM/Transformer models
  - `signals.py`: Signal generation
  - `risk.py`: Risk management
  - `monte_carlo.py`: Monte Carlo simulation
  - `backtest.py`: Backtesting engine
  - `visualization.py`: Plotting and reporting
  - `portfolio.py`: Portfolio management
- Place unit tests in `tests/unit/` and integration tests in `tests/integration/`.
- Use `data_cache/` for cached data, `plots/` for generated plots, and `output_conservative/` for outputs.

### Configuration & Parameters
- Use **dataclasses** for configuration objects.
- Validate configuration parameters in `__post_init__` methods.
- Store all configuration and hyperparameters in code or YAML files, not hardcoded in logic.

### Logging & Error Handling
- Use the **logging** module for all runtime messages; avoid print statements in production code.
- Raise exceptions for invalid parameters or errors; do not fail silently.

### Testing
- Use **pytest** for all tests.
- Place unit tests in `tests/unit/`, integration tests in `tests/integration/`.
- Ensure all new features are covered by tests.
- Check test coverage with `pytest-cov`.

### Documentation
- Use docstrings for all public classes, methods, and functions.
- Generate API documentation with **Sphinx**.
- Maintain a clear and up-to-date `README.md`.

### Notebooks
- Use Jupyter notebooks (e.g., `ProjectDemo.ipynb`) for demos and exploration only.
- Do not place core logic in notebooks.

### Dependency Management
- List all dependencies in `requirements.txt`.
- Use virtual environments for development.

### Pre-commit Hooks
- Use **pre-commit** to run formatting and linting checks before commits.
- Typical hooks: black, flake8, isort, mypy, autopep8.

### Data & Outputs
- Do not commit large data files, models, or outputs to version control.
- Use `.gitignore` to exclude data, models, and environment files.

### Version Control
- Use feature branches for new features or bug fixes.
- Ensure all tests pass before submitting a pull request.
- Write clear, descriptive commit messages.

### Miscellaneous
- Use environment variables or config files for secrets and credentials (never hardcode).
- Prefer vectorized operations with numpy/pandas over loops for performance.
- Clean up unused imports and variables.

---

_These rules are inferred from the codebase structure, requirements, and documentation. For any questions or to propose changes, update this file and notify the team._ 