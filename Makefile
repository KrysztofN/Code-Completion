.PHONY: dataset-code dataset-text prediction-code prediction-text charts 

mode ?= "PSM"

ifeq ($(OS),Windows_NT)
	PYTHON_CMD := python
else
	PYTHON_CMD := python3
endif

dataset-code: 
	$(PYTHON_CMD) code_completion.py --mode $(mode)

dataset-text: 
	$(PYTHON_CMD) text_completion.py --mode $(mode)

prediction-code: 
	$(PYTHON_CMD) code_fim_dataset_generator.py

prediction-text: 
	$(PYTHON_CMD) text_fim_dataset_generator.py

charts: 
	$(PYTHON_CMD) charts.py