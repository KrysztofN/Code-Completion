.PHONY: dataset-code dataset-text prediction-code prediction-text charts

mode ?= "PSM"

dataset-code:
	py code_completion.py --mode $(mode)

dataset-text:
	py text_completion.py --mode $(mode)

prediction-code:
	py code_fim_dataset_generator.py

prediction-text:
	py text_fim_dataset_generator.py 

charts:
	py charts.py 