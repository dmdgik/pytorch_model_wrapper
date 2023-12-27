# Getting started with pytorch model wrapper

## Installation

```console
pip3 install pytorch-model-wrapper
```

## What and how?

Essentially, a wrapper consists of the wrapper code itself and the config that is supplied to define this wrapper. The main task is to understand how the config works. Further, using the wrapper simply comes down to the usual fit-predict methods like in classic ML.

## Base config that the wrapper uses and how it can be changed

From the very beginning, it is better to consider the structure of the config that can be used to wrap the model. Below is the default base config used by the wrapper. For ease of visualization, the yaml config file will be considered, but essentially you can use a regular dictionary in python. The provided config in the form of a python dictionary can be found in [this file](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/src/pytorch_model_wrapper/fittable_model_wrapper_commons.py) in the class **FittableModelBaseConfig**

```yaml
init:
    experiment_name: 
    accelerator:
        mixed_precision: "no"
        gradient_accumulation_steps: 1
        step_scheduler_with_optimizer: false
        cpu: true
    logging_file:
    log_only_main_process: true
common:
    from_checkpoint:
    check_run_num:
dataloader:
    batch_size: 1
    shuffle: true
    pin_memory: false
optimizer:
    optimizer_name: "Adam"
    optimizer_params:
        lr: 0.001
scheduler:
    scheduler_name:
    scheduler_params: {}
monitor:
    monitor_enabled: false
    monitor_dir: "./"
checkpoint:
    saving_checkpoint_enabled: false
    save_checkpoint_every: 1
    save_torchscripted: false
    checkpoint_dir: "./"
    saving_without_optimizer_and_scheduler: false
    train_history_dir: "./"
snapshot:
    saving_snapshot_enabled: false
    save_snapshot_every: 1
    snapshot_dir: "./"
mlflow:
    saving_mlflow_checkpoint_enabled: false
    save_mlflow_checkpoint_every: 1
    save_mlflow_torchscripted: false
    mlflow_tracking_server_uri:
    mlflow_experiment_name:
    mlflow_tags: {}
    mlflow_params: {}
fit:
    fit_name:
    epochs: 1
    freezed_params_names: []
    clip_grad_value_:
    clip_grad_norm_:
    metrics: []
    overfitting_detector_enabled: false
    overfitting_detector_metric:
    overfitting_detector_epochs: 1
evaluate:
    save_evaluate_results: false
    evaluate_results_file: "./evaluate_results.yaml"
    metrics: []
predict:
    predict_results_file: "./predict_results.pt"
s3_uploading:
    bucket_name:
    folder_name: ""
    save_model: false
    save_model_torchscripted: false
    save_fit_results: false
    save_logs: false
    save_monitor_logs: false
    save_evaluate_results: false
    save_predict_results: false
extra: {}
```

Let's look at all parts of the config in order.

1. <span style="color: green;">**init**</span> - Part of the config that cannot be changed after the wrapper class object is created.

```yaml
init:
    experiment_name:
    accelerator:
        mixed_precision: "no"
        gradient_accumulation_steps: 1
        step_scheduler_with_optimizer: false
        cpu: true
    logging_file:
    log_only_main_process: true
```

- **experiment_name:** The name of your experiment with the model. 
<br><br> *Data type - string or empty. Default empty (None for python). If you leave this field empty, as a result, when processing the config, experiment_name will become the string "None"*

- **accelerator:** Huggingface Accelerator parameters. You can read about each of the parameters in the Huggingface Accelerate documentation. **NOTE:** The wrapper was tested and written for use on CPUs and CUDA GPUs. Using a wrapper on TPU will not work.

    - **mixed_precision:** The ability to set the precision of model parameters to speed up and distillation the model. 
    <br><br> *Data type - string. Default value "no".*

    - **gradient_accumulation_steps:** Number of steps for gradient accumulation during training. 
    <br><br> *Data type - int. Default value 1.*

    - **step_scheduler_with_optimizer:** In the current version of the wrapper, the scheduler can be set in steps only by epoch, so this value must be left false (in the future, the ability to use the scheduler by batch may be added). 
    <br><br> *Data type - bool. Default value false.*

    - **cpu:** Enforcement of script execution on the CPU. 
    <br><br> *Data type - bool. Default value true. **Note:** When running on a device without cuda, it is better set it true.*

- **logging_file:** File path for saving logs. 
<br><br> *Data type - string or empty. Default empty (None for python). If the field is left empty, the logs will not be saved anywhere.*

- **log_only_main_process:** Is responsible for logging information from the main process. If you use multiple GPUs, logging will be from the main GPU device. If you set it false, then logging will be from all parallelized devices. 
<br><br> *Data type - bool. Default value true.*
    
2. <span style="color: green;">**common**</span> - Part of the config that is used in the fit, evaluate and predict methods. Determines whether it is necessary to start from any checkpoint and the number of iterations of the dataloader (for testing your pipeline code if it is necessary).

```yaml
common:
    from_checkpoint:
    check_run_num:
```

- **from_checkpoint:** File path of checkpoint from which the model will be loaded. The **checkpoint** file must contain a dictionary with keys **"model_state_dict"**, **"optimizer_state_dict"**, **"sceduler_state_dict"**. According to the "model_state_dict" key the model state dict should be, according to the "optimizer_state_dict" key the optimizer state dict should be and according to the "sceduler_state_dict" key the scheduler state dict should be. **"optimizer_state_dict"** and **"sceduler_state_dict"** **may be None**. 
<br><br> *Data type - string or empty. Default empty (None for python). If you leave the field empty no checkpoint will be loaded.*

- **check_run_num:** Number of dataloader iterations. Made for using code debugging, so as not to wait for the entire dataloader to run. 
<br><br> *Data type - int or empty. Default empty (None for python). If you leave the field empty the dataloader will run completely.*

3. <span style="color: green;">**dataloader**</span> - Part of the config that defines the pytorch DataLoader parameters.

```yaml
dataloader:
    batch_size: 1
    shuffle: true
    pin_memory: false
```

- **batch_size:** DataLoader batch size.
<br><br> *Data type - int. Default 1.*

- **shuffle:** DataLoader shuffle.
<br><br> *Data type - bool. Default true.*

- **pin_memory:** DataLoader pin memory.
<br><br> *Data type - bool. Default false.*

4. <span style="color: green;">**optimizer**</span> - Part of the config that defines the optimizer.

```yaml
optimizer:
    optimizer_name: "Adam"
    optimizer_params:
        lr: 0.001
```

- **optimizer_name:** Optimizer name in **FittableModelAvailables.optimizers**.
<br><br> *Data type - string. Default "Adam".*

- **optimizer_params:** Dictionary of optimizer hyperparameters that will be used to create it. Supplied to the optimizer as kwargs **optimizer_params.
<br><br> *Data type - dict. Default {"lr" : 0.001}.*

5. <span style="color: green;">**scheduler**</span> - Part of the config that defines the scheduler.

```yaml
scheduler:
    scheduler_name:
    scheduler_params: {}
```

- **scheduler_name:** Scheduler name in **FittableModelAvailables.schedulers**. 
**NOTE** In the current version, the scheduler can only make a step after each epoch.
<br><br> *Data type - string or empty. Default empty. If empty, then the scheduler will not be created*.

- **scheduler_params:** Dictionary of scheduler hyperparameters that will be used to create it. Supplied to the scheduler as kwargs **scheduler_params.
<br><br> *Data type - dict. Default {}.*

6. <span style="color: green;">**monitor**</span> - Part of the config that defines the tensorboard monitor.

```yaml
monitor:
    monitor_enabled: false
    monitor_dir: "./"
```

- **monitor_enabled:** Will the tensorboard monitor be turned on?
<br><br> *Data type - bool. Default false.*

- **monitor_dir:** The directory where the tensorboard logging results will be saved. **NOTE:** for each launch of fit, a subdirectory called "yourexperimentname_yourfitname" will be created in the root monitor directory.
<br><br> *Data type - string. Default "./".*

7. <span style="color: green;">**checkpoint**</span> - Part of the config that defines checkpointing. The checkpoint will be saved in dictionary format with the keys "model_state_dict", "optimizer_state_dict", "scheduler_state_dict" and "fittable_config" in a file with the name "checkpoint_yourexperimentname_yourfitname.pt". In "fittable_config" the entire result config dictionary will be saved with additional key "fit_results", where the results obtained on the train and valid datasets, running epoch, used device, num processes information will be saved. If the overfitting detector is enabled, only the best version of the model will be saved. Also, if the overfitting detector is enabled, then the save will be performed every epoch if the model is the best, and the **save_checkpoint_every** field will be ignored. If saving the torchscripted version of the model is enabled, it will be saved in the file "checkpoint_yourexperimentname_yourfitname_torchscripted.pt". 

```yaml
checkpoint:
    saving_checkpoint_enabled: false
    save_checkpoint_every: 1
    save_torchscripted: false
    checkpoint_dir: "./"
    saving_without_optimizer_and_scheduler: false
    train_history_dir: "./"
```

- **saving_checkpoint_enabled:** Will saving checkpoints be enabled?
<br><br> *Data type - bool. Default false.*

- **save_checkpoint_every:** Once in how many epochs will a checkpoint be saved? Also, the checkpoint of the last epoch will be saved (except when the overfitting detector is enabled - then only the best model is saved).
<br><br> *Data type - int. Default 1.*

- **save_torchscripted:** Will the torchscripted checkpoint be saved?
<br><br> *Data type - bool. Default false.*

- **checkpoint_dir:** Directory where checkpoints will be saved.
<br><br> *Data type - string. Default "./".*

- **saving_without_optimizer_and_scheduler:** Flag for saving checkpoint without optimizer state dict and scheduler state dict. In the checkpoint file in the dictionary by keys "optimizer_state_dict" and "scheduler_state_dict" there will be None values.
<br><br> *Data type - bool. Default false.*

- **train_history_dir:** Directory for saving a file with model training history by epoch (including train losses at each step, train metrics and valid metrics). File name will be "epochs_results_yourexperimentname_yourfitname.yaml"
<br><br> *Data type - string. Default "./".*

8. <span style="color: green;">**snapshot**</span> - Part of the config that defines the snapshotting. Snapshotting is used to restore the training process when the training process is unexpectedly interrupted. All necessary data for restoring the training process will be in the snapshot file. When you restart the training script after an unexpected interruption, the wrapper itself will load everything it needs from the snapshot file, if one is found. Snapshot file name "snapshot_yourfitname.pt". After successful completion of the experiment, it is recommended to delete snapshots as unnecessary. For this purpose, the wrapper provides a special method called **clean_snapshots**, but more on that later.

```yaml
snapshot:
    saving_snapshot_enabled: false
    save_snapshot_every: 1
    snapshot_dir: "./"
```

- **saving_snapshot_enabled:** Will snapshots be saved?
<br><br> *Data type - bool. Default false.*

- **save_snapshot_every:** Once in how many epochs will a snapshot be saved?
<br><br> *Data type - int. Default 1.*

- **snapshot_dir:** The directory for snapshot files store.
<br><br> *Data type - string. Default "./".*

9. <span style="color: green;">**mlflow**</span> - Part of the config that defines the mlflow. Inside one mlflow checkpoint there will be: a dictionary with train losses, train and valid metrics in file "train_results.yaml"; dictionary of passed config in file fittable_config.yaml; model in artifact "model"; model torchscripted in artifact "model_scripted". 

<br> Default list of tags: 
- "experiment_name" // config init experiment_name
- "optimizer_name" // config optimizer_name
- "scheduler_name" // config scheduler_name 
- "accelerator_mixed_precision" // config accelerator mixed_precision
- "accelerator_cpu" // config accelerator cpu
- "save_mlflow_torchscripted" // config save_mlflow_torchscripted
- "from_checkpoint" // config from_checkpoint
- "overfitting_detector_enabled" // config overfitting_detector_enabled
- "overfitting_detector_metric" // config overfitting_detector_metric
- "best_model" // best checkpoint if model overfitted detector overfits. **NOTE:** regular checkpoint will save only best model by overfitted detector, but mlflow checkpoint will save all next steps untill overfitted detector becomes final ovefit. Finally best model will be saved in mlflow.
- "used_device" // used device for training "cuda" or "cuda:0" or "cpu"
<p> 

<br> Default list of params:
- "check_run_num" // config check_run_num
- "gradient_accumulation_steps" // config gradient_accumulation_steps
- "step_scheduler_with_optimizer" // config step_scheduler_with_optimizer
- "num_processes" // how many GPUs used for training
- "dataloader_batch_size" // config dataloader batch_size
- "dataloader_shuffle" // config dataloader shuffle
- "dataloader_pin_memory" // config dataloader pin_memory
- "optimizer_paramname" for paramname in optimizer_params // every parameter determines in config optimizer_params
- "scheduler_paramname" for paramname in scheduler_params // every parameter determines in config scheduler_params
- "clip_grad_value_" // config clip_grad_value_
- "clip_grad_norm_" // config clip_grad_norm_
- "overfitting_detector_epochs" // config overfitting_detector_epochs
- "running_epoch" // current running epoch
- "model_parametrs_count" // model parameters count
- "model_fittable_parametrs_count" // model trainable parameterts count
<p>

<br> List of metrics:
- "train_metricname" for metricname in train_metrics // every metric calculated for train dataset
- "valid_metricname" for metricname in valid_metrics // every metric calculated for valid dataset

```yaml
mlflow:
    saving_mlflow_checkpoint_enabled: false
    save_mlflow_checkpoint_every: 1
    save_mlflow_torchscripted: false
    mlflow_tracking_server_uri:
    mlflow_experiment_name:
    mlflow_tags: {}
    mlflow_params: {}
```

- **saving_mlflow_checkpoint_enabled:** Will mlflow be enabled?
<br><br> *Data type - bool. Default false.*

- **save_snapshot_every:** Once in how many epochs will a mlflow checkpoint be saved?
<br><br> *Data type - int. Default 1.*

- **save_torchscripted:** Will the torchscripted model in mlflow checkpoint be saved?
<br><br> *Data type - bool. Default false.*

- **mlflow_tracking_server_uri:** Mlflow tracking server uri.
<br><br> *Data type - string or empty. Default empty.*

- **mlflow_experiment_name:** Mlflow experiment name.
<br><br> *Data type - string or empty. Default empty. If mlflow is enabled and the **"mlflow_experiment_name"** is empty, then the value will become "None"*.

- **mlflow_tags:** Additional tags for saving in mlflow. Dictionary format - {tag_name: tag_value}.
<br><br> *Data type - dict. Default {}.*

- **mlflow_params:** Additional params for saving in mlflow. Dictionary format - {param_name: param_value}.
<br><br> *Data type - dict. Default {}.*

10. <span style="color: green;">**fit**</span> - Part of the config that defines the fit parameters.

```yaml
fit:
    fit_name:
    epochs: 1
    freezed_params_names: []
    clip_grad_value_:
    clip_grad_norm_:
    metrics: []
    overfitting_detector_enabled: false
    overfitting_detector_metric:
    overfitting_detector_epochs: 1
```

- **fit_name:** Name of your fit.
<br><br> *Data type - string or empty. Default empty. If empty it will converts to "None".*

- **epochs:** Count of epochs for your fit.
<br><br> *Data type - int. Default 1.*

- **freezed_params_names:** Model's parameters names which will be freezed during training. Can be used for fine tuning pretrained model.
<br><br> *Data type - list. Default [].*

- **clip_grad_value_:** Gradient clipping for preventing gradient explosion.
<br><br> *Data type - float or empty. Default empty. If empty it will not uses in model training process.*

- **clip_grad_norm_:** Gradient norm scaling for preventing gradient explosion.
<br><br> *Data type - float or empty. Default empty. If empty it will not uses in model training process.*

- **metrics:** List of metrics you want to calculate at the end of epoch. Must be in **FittableModelAvailables.metrics**. For train and valid datasets by default will be calculated criterion. For train dataset also will be calculated train time by default.
<br><br> *Data type - list. Default [].*

- **overfitting_detector_enabled:** Will there be an overfitting detector enabled? If yes, then you must pass a validation dataset to fit method. The overfitting detector will look at the validation dataset.
<br><br> *Data type - bool. Default false.*

- **overfitting_detector_metric:** Metric that the overfitting detector will look at. 
<br><br> *Data type - string or empty. Default empty. If empty and overfitting detector enabled it will look at criterion.*

- **overfitting_detector_epochs:** How many more epochs will be run to give the model a chance to get a better result if the model is overfiited? For example, if this value is 1, then as soon as the validation dataset turns out to have a metric value worse than the previous epoch, the model training process will stop.
<br><br> *Data type - int. Default 1.*

11. <span style="color: green;">**evaluate**</span> - Part of the config that defines the evaluate parameters.

```yaml
evaluate:
    save_evaluate_results: false
    evaluate_results_file: "./evaluate_results.yaml"
    metrics: []
```

- **save_evaluate_results:** Will the results be saved to a file?
<br><br> *Data type - bool. Default false. If false, the results will only be displayed in the logs.*

- **evaluate_results_file:** File path for saving evaluating results.
<br><br> *Data type - string. Default "./evaluate_results.yaml".*

- **metrics:** List of metrics you want to calculate for evaluate dataset. Must be in **FittableModelAvailables.metrics**.
<br><br> *Data type - list. Default [].*

12. <span style="color: green;">**predict**</span> - Part of the config that defines the predict parameters. 

```yaml
predict:
    predict_results_file: "./predict_results.pt"
```

- **evaluate_results_file:** File path for saving predict results. The file will contain a dictionary with the keys "ids" and "outputs" with the contents of the corresponding tensors.
<br><br> *Data type - string. Default "./predict_results.pt".*

13. <span style="color: green;">**s3_uploading**</span> - Part of the config that defines the AWS S3 bucket uploading parameters. To use the method that uses this part of the config, it is necessary that the environment contains variables for the AWS keys with the names AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY and the corresponding values of your keys for AWS.

```yaml
s3_uploading:
    bucket_name:
    folder_name: ""
    save_model: false
    save_model_torchscripted: false
    save_fit_results: false
    save_logs: false
    save_monitor_logs: false
    save_evaluate_results: false
    save_predict_results: false
```

- **bucket_name:** AWS bucket name.
<br><br> *Data type - string or empty. Default empty. If you are going to use the method to upload results to a AWS S3 bucket, then you must specify the name of the bucket*

- **folder_name:** AWS bucket folder name. The folder in your bucket where the results files will be uploaded.
<br><br> *Data type - string. Default "".*

- **save_model:** If there is a file with a model, will it be saved in s3?
<br><br> *Data type - bool. Default false.*

- **save_model_torchscripted:** If there is a file with torchscripted model, will it be saved in s3?
<br><br> *Data type - bool. Default false.*

- **save_fit_results:** If there is a file with fit results, will it be saved in s3?
<br><br> *Data type - bool. Default false.*

- **save_logs:** If there is a file with logs, will it be saved in s3?
<br><br> *Data type - bool. Default false.*

- **save_monitor_logs:** If there is a file with tensorboard monitor logs, will it be saved in s3?
<br><br> *Data type - bool. Default false.*

- **save_evaluate_results:** If there is a file with evaluate results (provided in config), will it be saved in s3?
<br><br> *Data type - bool. Default false.*

- **save_predict_results:** If there is a file with predict results (provided in config), will it be saved in s3?
<br><br> *Data type - bool. Default false.*

14. <span style="color: green;">**extra**</span> - Part of the config in which you can save any extra information you need in dictionary format.

```yaml
extra: {}
```

## **!!! IMPORTANT !!!** Required dataset and model input format

The dataset must return three variables:

1. Tensor with IDs of dataset objects
2. Dict of features in format {feature_name: feature_tensor}
3. Tensor with targets

Fit and evaluate methods will not pay attention to the tensor with IDs (you can fill it with empty tensor if necessary), predict method will not pay attention to the tensor with targets (you can fill it with empty tensor if necessary). But at the moment, in order for everything to work seamlessly and to be able to use the predict method, it is necessary that the tensor with IDs be returned (since when using multiple GPUs for prediction, the dataloader will in any case shuffle up the data (even if you put a dataloader shuffle parametr false) and it will not be possible to easily restore the initial order of the dataset objects). 

Since the dataset must return features as a dictionary with tensors, the model must also accept a dictionary with tensors as input.

In general, taking into account the fact that now quite often inputs are supplied to models that are not one tensor, but a set of tensors, they can also be considered as a certain set of hyperfeatures :smile:

## FittableModelWrapper and FittableModelAvailables

Finally we got to the main wrapper classes.
<p>

### FittableModelWrapper

FittableModelWrapper - class that allows you to wrap the pytorch model and use the above config.

1. initialization

To create an object of the FittableModelWrapper class, you need to pass your pytorch model and optionally a dictionary with your config (if you do not pass the dictionary of the config, the standard base config discussed above will be used). You don't have to define every component of the config to pass it (you can only define the necessary parts that you want to change).

```python
FittableModelWrapper(model: torch.nn.Module, fittable_model_config: Optional[dict] = None) # model - your pytorch model, fittable_model_config - your changes to default fittable config
```

```python
from pytorch_model_wrapper import FittableModelWrapper
from your_model_python_file import YourModel

your_config_dict = {...}
model = YourModel()

model = FittableModelWrapper(model, your_config_dict)
```

2. methods

The following methods are now available for the wrapped model:

```python
fit(criterion: Any, dataset_train: Dataset, dataset_valid: Optional[Dataset] = None, config_dict: Optional[dict] = None) # criterion - is your loss criterion, dataset_train - train dataset, dataset_valid - validation dataset if necessary, config_dict - dict with updates for your current config if necessary. Runs fit process.

evaluate(criterion: Any, dataset: Dataset, config_dict: Optional[dict] = None) # criterion - is your loss criterion, dataset - dataset for evaluating, config_dict - dict with updates for your current config if necessary. Runs evaluate process.

predict(dataset: Dataset, config_dict: Optional[dict] = None) # dataset - dataset for predicting, config_dict - dict with updates for your current config if necessary. Runs predict process.

clean_snapshots() # Removes snapshot files.

s3_upload(config_dict: Optional[dict] = None) # config_dict - dict with updates for your current config if necessary. Uploads files in AWS S3
```

Little python example: 

```python
from pytorch_model_wrapper import FittableModelWrapper
from your_model_python_file import YourModel
from your_dataset_python_file import YourDatset
import torch

your_config_dict = {...}
your_config_dict_update_1 = {...}
your_config_dict_update_1_1 = {...}
your_config_dict_update_2 = {...}

model = YourModel()
criterion = torch.nn.CrossEntropyLoss()
dataset_train = YourDatset("train")
dataset_valid = YourDatset("valid")
dataset_test = YourDatset("test")

model = FittableModelWrapper(model, your_config_dict)
model.fit(criterion, dataset_train, dataset_valid)
model.evaluate(criterion, dataset_test)
model.predict(dataset_test)
model.predict(dataset_valid, your_config_dict_update_1) # for example we want to save model outputs also for dataset_valid and in your_config_dict_update_1 we changed predict_results_file
model.s3_upload()
model.s3_upload(your_config_dict_update_1_1) # in previous s3 uploading predict file for dataset_test will not upload as we changed the config. Now we passing predict_results_file which was in initial config and trying to upload only this predicts file.
model.fit(criterion, dataset_train, dataset_valid, your_config_dict_update_2) # for example we unfreezed some parameters of model and run new fit
model.evaluate(criterion, dataset_test)
model.predict(dataset_test)
model.s3_upload()
model.clean_snapshots() # removes all snapshots of the model if they stored in one directory (if you are changing config for new fits it is better to not change snapshot dir)
```

### FittableModelAvailables

FittableModelAvailables - class in which you can define dictionaries with used optimizers, schedulers and metrics

FittableModelWrapper while creating optimizers, schedulers and metrics will look at FittableModelAvailables to find necessary optimizer by optimizer_name, scheduler by scheduler_name and metric by metric_name in list of metrics passed in config.

By default 
```python
FittableModelAvailables.optimizers # {"Adam": torch.optim.Adam}
FittableModelAvailables.schedulers # {}
FittableModelAvailables.metrics # {}
``` 

Let's look how to set your optimizers, schedulers and metrics in FittableModelAvailables.

```python
from pytorch_model_wrapper import FittableModelAvailables
from your_metrics_file import your_metric1, your_metric2
import torch

FittableModelAvailables.optimizers["AdamW"] = torch.optim.AdamW
FittableModelAvailables.schedulers["StepLR"] = torch.optim.lr_scheduler.StepLR
FittableModelAvailables.metrics["your_metric1"] = [your_metric1, 1]
FittableModelAvailables.metrics["your_metric2"] = [your_metric2, -1]
```

For FittableModelAvailables.metrics you must pass a list with your metric function, which return one float value, and direction of better metric values. Last thing uses in overfitting detector. If your metric is better when it is larger values, than pass 1. If your metric is better when it is smaller values, than pass -1. For example accuracy and rmse metrics:

```python
from pytorch_model_wrapper import FittableModelAvailables
from your_metrics_file import accuracy, rmse

FittableModelAvailables.metrics["accuracy"] = [accuracy, 1]
FittableModelAvailables.metrics["rmse"] = [rmse, -1]
```

## Simple Example

Full code examples explanations you can find [here](https://github.com/dmdgik/pytorch_model_wrapper/blob/main/docs/scripts_examples.md)

Below is an example of python pseudocode for executing the script

train_script.py :
```python
from pytorch_model_wrapper import FittableModelAvailables
from pytorch_model_wrapper import FittableModelWrapper
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

As pytorch_model_wrapper uses accelerate, then launch of this script will be with this command (after configuring accelerate config): 

```console
accelerate launch train_script.py
```

