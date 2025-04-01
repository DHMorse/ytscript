# 🚀 Environment Setup Guide

## Prerequisites
- Python 3.10 or higher
- Make

## 🔧 Installation Options

### Option 1: Using UV (Recommended)

1. **Check UV installation**
    ```bash
    uv --version
    ```

2. **Sync dependencies**
    ```bash
    uv sync
    ```

3. **Activate virtual environment**
    ```bash
    source .venv/bin/activate
    ```

### Option 2: Using Pip

1. **Create virtual environment**
    ```bash
    python3 -m venv .venv
    ```

2. **Activate virtual environment**
    ```bash
    source .venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏗️ Build and Run

### Build
```bash
make build
```

### Run
```bash
make run
```