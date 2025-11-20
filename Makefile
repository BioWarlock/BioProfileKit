define help

Supported targets: 'install' 'clean' 'reinstall' 'build-docker' 'docker-test'
The 'install' target installs BioProfileKit build requirements into the current virutalenv.
The 'clean' target undoes the effect of 'install'
The 'reinstall' target runs the 'clean' and 'install' target
The 'build-docker' target runs docker compose


endef
export help
help:
	@echo "$$help"

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

.PHONY: install clean reinstall help build-docker docker-test