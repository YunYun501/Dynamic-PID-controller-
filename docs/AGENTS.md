# AGENTS.md

## Build/Lint/Test Commands
- Run simulation: `python run_sim.py`
- Run with specific mode: `python run_sim.py --mode position` (or velocity, acceleration)
- No dedicated test suite available
- Lint with: `ruff check .` (if installed)
- Format with: `ruff format .` (if installed)

## Code Style Guidelines
- Use Python 3.11+ with type hints
- Follow PEP 8 for formatting
- Use numpy-style docstrings
- Imports: Standard library, third-party, local (alphabetized)
- Naming: snake_case for variables/functions, PascalCase for classes
- Error handling: Use appropriate exceptions, avoid generic except
- Types: Use explicit type hints for function parameters and returns
- Comments: Explain why, not what; focus on complex logic
- Constants: Use UPPER_SNAKE_CASE for module-level constants

## Environment Details
- Custom Gymnasium environment in modular structure under `environment/` directory
- Three control modes: position, velocity, and acceleration control
- Uses pygame for rendering with T-shaped joint visualization
- CSV logging and matplotlib plotting in `run_sim.py`

## Special Notes
- No .cursorrules or .github/copilot-instructions.md files found
- Project focuses on soft robotic arm control with dynamic tuning