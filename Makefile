.PHONY: quality style

# Check that source code meets quality standards

quality:
	black --check .
	isort --check-only .
	flake8 .

# Format source code automatically

style:
	black .
	isort .
