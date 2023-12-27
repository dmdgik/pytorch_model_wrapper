# Pytorch model wrapper

## Make your life easier with a special wrapper for Pytorch models.

Solving any problem using ML essentially means specifying a complex set of hyperparameters for training the model. In fact, the model itself and the dataset on which it is trained are also hyperparameters of the general ML task. To run any experiments related to ML tasks, we want to set and save everything related to the experiment in the simplest way. That is, as a result, any ML experiment can be reduced to a certain config in which the hyperparameters of the experiment will be determined.

Also, sometimes you want to use the capabilities of logging experiments in mlflow, use a tensorboard to track the learning process of a model online, save the model itself for additional training or for implementation in production via a torchscript, use the ability to train on several GPUs and not only by using the huggingface accelerate library and much more.

It would also be nice to be able to start the process of training pytorch neural networks, as this is done in classical ML libraries like sklearn, XGBoost and CATBoost, by simply calling the fit method, saving predictions to a file by calling only the predict method, or getting metrics results on a new dataset by calling evaluate method (this is already an addition, it is especially useful for various competitions such as Kaggle)

All of the above and more were implemented in the pytorch model wrapper package. Let's see how it works.

- [Getting started](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/docs/getting_started.md)
- [Scripts examples](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/docs/scripts_examples.md)


## The most important thing is what can be specified in the wrapper?

- huggingface accelerator parametrs (Yes, you can use multigpu training, evaluating and predicting. Mixed precision, gradient accumulation steps and CPU enforcement are also available)
- file for saving logs (Store history of your wrapper work)
- upload checkpoint file (Do you want to continue training from the checkpoint? No problem. Or do you want to get metrics results on new data from a specific checkpoint? Not a problem either. Do you want to get model outputs using a specific checkpoint? And it's possible)
- dataloader parameters (batch_size, shuffle, pin_memory)
- number of iterations of the dataloader (For the possibility of a test run to check the functionality of the script)
- name and parameters of the optimizer
- name and parameters of the sceduler
- recording monitoring files using tensorboard
- saving checkpoints and training history
- saving snapshots for the possibility of restoring the training process in case of unexpected interruption
- saving data in mlflow
- fit parameters - number of epochs, names of parameters for freezing (you can fine tune the model), clipping gradients, calculation of custom metrics, overfitting detector with the ability to specify a metric by which to control overfitting and the number of overfitted epochs
- evaluate metrics and saving them to a file
- predictions save file
- s3 bucket uploading

## If the capabilities of the wrapper interest you, then I suggest you take a look at the documentation and examples of use.

- [Getting started](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/docs/getting_started.md)
- [Scripts examples](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/docs/scripts_examples.md)

## Usage example

1. Installation

```console
pip3 install pytorch-model-wrapper
```

2. Import

```python
from pytorch_model_wrapper import FittableModelWrapper, FittableModelAvailables
```

3. Simple full config example python dictionary, simple full [config in yaml](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/examples/configs/fittable_model_base_config.yaml)

```python
simple_config = {
    "init": {
        "experiment_name": None,
        "accelerator": {
            "mixed_precision": "no",
            "gradient_accumulation_steps": 1,
            "step_scheduler_with_optimizer": False,
            "cpu": True,
        },
        "logging_file": None,
        "log_only_main_process": True,
    },
    "common": {
        "from_checkpoint": None,
        "check_run_num": None,
    },
    "dataloader": {
        "batch_size": 1,
        "shuffle": True,
        "pin_memory": False,
    },
    "optimizer": {
        "optimizer_name": "Adam",
        "optimizer_params": {"lr": 0.001},
    },
    "scheduler": {
        "scheduler_name": None,
        "scheduler_params": {},
    },
    "monitor": {"monitor_enabled": False, "monitor_dir": "./"},
    "checkpoint": {
        "saving_checkpoint_enabled": False,
        "save_checkpoint_every": 1,
        "save_torchscripted": False,
        "checkpoint_dir": "./",
        "saving_without_optimizer_and_scheduler": False,
        "train_history_dir": "./",
    },
    "snapshot": {
        "saving_snapshot_enabled": False,
        "save_snapshot_every": 1,
        "snapshot_dir": "./",
    },
    "mlflow": {
        "saving_mlflow_checkpoint_enabled": False,
        "save_mlflow_checkpoint_every": 1,
        "save_mlflow_torchscripted": False,
        "mlflow_tracking_server_uri": None,
        "mlflow_experiment_name": None,
        "mlflow_tags": {},
        "mlflow_params": {},
    },
    "fit": {
        "fit_name": None,
        "epochs": 1,
        "freezed_params_names": [],
        "clip_grad_value_": None,
        "clip_grad_norm_": None,
        "metrics": [],
        "overfitting_detector_enabled": False,
        "overfitting_detector_metric": None,
        "overfitting_detector_epochs": 1,
    },
    "evaluate": {
        "save_evaluate_results": False,
        "evaluate_results_file": "./evaluate_results.yaml",
        "metrics": [],
    },
    "predict": {
        "predict_results_file": "./predict_results.pt",
    },
    "s3_uploading": {
        "bucket_name": None,
        "folder_name": "",
        "save_model": False,
        "save_model_torchscripted": False,
        "save_fit_results": False,
        "save_logs": False,
        "save_monitor_logs": False,
        "save_evaluate_results": False,
        "save_predict_results": False,
    },
    "extra": {},
}
```

4. Train script example

train_script.py:
```python
from pytorch_model_wrapper import FittableModelWrapper, FittableModelAvailables
from your_model_python_file import YourModel # some your classification model which can use dict of tensors as input
from your_dataset_python_file import YourDatset # your custom dataset, which returns tensor with ids, feature dict with tensors and tensor with targets
import torch

def accuracy_score(outputs, targets):
    return torch.sum(torch.argmax(outputs, dim=1) == targets).item() / len(outputs)

FittableModelAvailables.metrics["accuracy"] = [accuracy_score, 1]
your_config_dict = {...} # now here in metrics you can use accuracy and overfitting detector by accuracy

model = YourModel()
criterion = torch.nn.CrossEntropyLoss()
dataset_train = YourDatset("train")
dataset_valid = YourDatset("valid")
dataset_test = YourDatset("test")

model = FittableModelWrapper(model, your_config_dict)
model.fit(criterion, dataset_train, dataset_valid)
model.evaluate(criterion, dataset_test)
model.predict(dataset_test)
model.s3_upload()
model.clean_snapshots()
```

5. run command

```console
accelerate launch train_script.py
```