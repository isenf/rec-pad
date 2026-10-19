# Pattern Recognition

## Prerequisites

Before you begin, make sure you have `uv` installed. `uv` will automatically download and manage the correct Python version of this project.

### Installing `uv`

If you don't have `uv` installed, you can install it using the following methods:

Linux/MacOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip:
```bash
pip install uv
```

## Setup and Environment Configuration

1. **Clone the repository:**

```bash
git@github.com:isenf/rec-pad.git
cd rec-pad
```

2. **Create the virtual environment and install dependencies:**

```bash
uv sync
```

3. **Activate the virtual environment (optional):**

For Linux/macOS:
```bash
source .venv/bin/activate
```

For Windows:
```cmd
.venv\Scripts\activate.bat
```

## Running the project

Run the main script using `uv`:
```bash
uv run src/rec_pad/main.py
```
