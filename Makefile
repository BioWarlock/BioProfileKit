.PHONY: install
install:
	pip install -e .
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf build/ dist/ */*.egg-info/
	find . -name ".so" -delete
	find . -name ".c" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

.PHONY: reinstall
reinstall: clean install