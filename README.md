# handwriting-recognition
Handwriting Recognition deep learning model using Kaggle dataset:

https://www.kaggle.com/datasets/landlord/handwriting-recognition/data

# Quickstart Guide

### Install poetry
We use `poetry` for dependency management, refer to [their documentation](https://python-poetry.org/docs/) for a thorough intro and installation guide.

If you install `poetry` with their official installer as opposed to `pipx`, make sure it's been added to your system path.

**Test** your installation by running the following command in a fresh shell:
```bash
poetry --version
```

### Install project dependencies

Simply run:
```
poetry install
```

### Install pre-commit hooks
We use `pre-commit` to ensure code style remains consistent across the repository. Before making any changes, run the following commands to install the `pre-commit` hooks in your local `.git` folder.

```bash
poetry run pre-commit install
```

Then use the following command to run the hooks on all files in the repo and initialise the environment:

```bash
poetry run pre-commit run --all-files
```

Now whenever you invoke `git commit`, the pre-commit hooks will lint your code and make it consistent with the style of the repo.

### Download the dataset

1) Create an account on https://www.kaggle.com/
2) Follow the instructions on https://www.kaggle.com/docs/api#authentication to create an API key for your Kaggle account
3) Run the following command:
```bash
poetry run python handwriting_recognition/data_downloader.py
```
