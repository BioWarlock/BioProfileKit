.PHONY: install clean reinstall help build-docker

help:
	@echo "BioProfileKit Commands:"
	@echo " make install		- Install package locally"
	@echo " make clean		    - Remove build files"
	@echo " make reinstall     - clean and install"

install:
	pip install -e .
	python setup.py build_ext --inplace

clean:
	rm -rf build/ dist/ */*.egg-info/
	find . -name ".so" -delete
	find . -name ".c" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

reinstall: clean install

build-docker:
	docker compose build

docker-test:
	docker compose run bioprofilekit -i data/winequality-white.csv