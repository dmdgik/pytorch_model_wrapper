# Examples

Code and other data for examples you can find [here](https://github.com/dmdgik/pytorch_model_wrapper/tree/main/examples)

## Data

Prepared [data](https://github.com/dmdgik/pytorch_model_wrapper/tree/main/examples/data) for examples with this simple python [script](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/data/create_data.py)

## Pytorch custom datasets

Custom datasets examples classes code [here](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/example_scripts/datasets.py)

## Models

Models examples you can find [here](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/example_scripts/models.py)

## Metrics

Custom metrics examples code [here](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/example_scripts/metrics.py)

## Experiment configs

Experiment [config](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/configs/regression.yaml) for regression

Experiment [config](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/configs/classification.yaml) for classification

**NOTE:** in extra part of config added some changes which will be passed to other parts of config inside the scripts.

## Scripts

Regression script example [here](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/example_scripts/regression.py) 

Classification script example [here](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/example_scripts/classification.py) 

## Scripts run commands

Go inside the example_scripts folder.

Classification:

```console
accelerate launch classification.py --experiment_config="../configs/classification.yaml"
```

Regression:

```console
accelerate launch regression.py --experiment_config="../configs/regression.yaml"
```

## Produced results

[Classification](https://github.com/dmdgik/pytorch_model_wrapper/tree/main/examples/results/classification)

[Regression](https://github.com/dmdgik/pytorch_model_wrapper/tree/main/examples/results/regression)