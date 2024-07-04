SHELL := /bin/bash
OUTPUT_DIR := output/
GIT_COMMIT_MSG := "Update experiment results"
GIT_BRANCH := main
PYTHON_SCRIPT := lazy.py
LLM := phi3-medium
DATASET1 := data/hpo/healthCloseIsses12mths0001-hard.csv
DATASET2 := data/hpo/healthCloseIsses12mths0011-easy.csv

.PHONY: help push run phi3

help:
	@echo "make phi3: Run the experiment using phi3-medium and store the result"

push: 
	git add $(OUTPUT_DIR)
	git commit -m "$(GIT_COMMIT_MSG)"
	git push origin $(GIT_BRANCH)

run:
	python3 $(PYTHON_SCRIPT) --llm $(LLM) --dataset $(DATASET1)
	python3 $(PYTHON_SCRIPT) --llm $(LLM) --dataset $(DATASET2)

phi3: run push
