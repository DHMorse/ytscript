# Building from Source

## Prerequisites

- Python 3.10 or higher
- Make
- Git
- pip or uv package manager

## Clone the Repository

```bash
git clone https://github.com/DHMorse/ytscript
cd ytscript
```

## Installation Methods

### Method 1: Using UV (Recommended)

UV offers faster dependency resolution and better reproducibility.

1. **Install UV** (if not already installed)
    ## Unix
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   ## Windows
   ```batch
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Create and activate virtual environment**
    ```bash
   uv venv
   ```
   ## Unix
   ```bash
   source .venv/bin/activate  
   ```
   ## Windows
   ```batch
   .venv\Scripts\activate     
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

### Method 2: Using Pip

1. **Create virtual environment**
   ## Unix
   ```bash
   python3 -m venv .venv
   ```
   ```bash
   source .venv/bin/activate 
   ```
   ## Windows
   ```batch
   python -m venv .venv
   ```
   ```batch
   .venv\Scripts\activate     
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Build and Run

### Build
```bash
make build
```

### Run the Application
```bash
make run
```

## Development Tools

### Run Tests
```bash
make test
```

## Troubleshooting

If you encounter build issues:

1. Ensure all prerequisites are installed
2. Check Python version compatibility
3. Clear previous build artifacts: `make clean`
4. Verify virtual environment activation

For more help, please open an issue on our GitHub repository.