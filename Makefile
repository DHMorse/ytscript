build:
	python -m nuitka --onefile --standalone --remove-output \
		--output-dir=dist --output-filename=ytscript \
		--nofollow-import-to=numba,triton \
		--module-parameter=torch-disable-jit=yes \
		--include-distribution-metadata=tiktoken --lto=yes \
		ytscript/main.py

run:
	dist/ytscript

test:
	python -m pytest tests/ -v

debug:
	python -m nuitka --onefile --standalone --remove-output \
		--output-dir=dist --output-filename=ytscript \
		--nofollow-import-to=numba,triton \
		--module-parameter=torch-disable-jit=yes \
		--include-distribution-metadata=tiktoken --lto=yes \
		--debug --verbose \
		ytscript/main.py > build.log

clean:
	rm -rf dist build build.log
	rm -rf *.txt
	rm -rf *.mp3
	rm -rf output/*
	uv pip freeze > requirements.txt

rebuild: 
	make clean
	make build

help: 
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  build: Build the project"
	@echo "  run: Run the project"
	@echo "  test: Run the tests"
	@echo "  debug: Build the project with verbose output"
	@echo "  clean: Clean the project"
	@echo "  rebuild: Clean and build the project"
	@echo "  help: Show this help message"
