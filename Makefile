# Default environment name and Python version
ENV_NAME ?= oreilly-llama3
PYTHON_VERSION ?= 3.10

repo-setup:
	mkdir -p requirements
	touch requirements/requirements.in
	echo jupyter >> requirements/requirements.in
	echo "Requirements folder created with jupyter package"

# Install exact Python and CUDA versions
conda-create:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION)
	echo "Conda environment $(ENV_NAME) created with Python $(PYTHON_VERSION)"

# Bump versions of transitive dependencies
pip-tools-setup:
	pip install uv
	uv pip install pip-tools setuptools

env-update:
	conda run -n $(ENV_NAME) bash -c "\
		uv pip compile ./requirements/requirements.in -o ./requirements/requirements.txt && \
		uv pip sync ./requirements/requirements.txt"

notebook-setup:
	python -m ipykernel install --user --name=$(ENV_NAME)

# Repo specific command
repo-specific-command:
	echo "Implement"

# Arcane incantation to print all the other targets
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
