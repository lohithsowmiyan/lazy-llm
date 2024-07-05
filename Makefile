SHELL := /bin/bash
OUTPUT_DIR := output
GIT_COMMIT_MSG := "Update experiment results"
GIT_BRANCH := main
PYTHON_SCRIPT := lazy.py
LLM := phi3-mini
DATASET1 := data/hpo/healthCloseIsses12mths0001-hard.csv
DATASET2 := data/hpo/healthCloseIsses12mths0011-easy.csv
DATASET3 := data/misc/auto93.csv

.PHONY: help push run phi3

help:
	@echo "make run: Run the experiment using any model and store the result"

push: 
	git add $(OUTPUT_DIR)
	git commit -m "$(GIT_COMMIT_MSG)"
	git push

run1:
	python3 $(PYTHON_SCRIPT) --llm $(LLM) --dataset $(DATASET1)
	
run2:
	python3 $(PYTHON_SCRIPT) --llm $(LLM) --dataset $(DATASET2)

run3:
	python3 $(PYTHON_SCRIPT) --llm $(LLM) --dataset $(DATASET3)

run: run1 push run2 push run3 push
