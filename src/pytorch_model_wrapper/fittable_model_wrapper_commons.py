import torch


class FittableModelBaseConfig:
    def __init__(self) -> None:
        """FittableModelBaseConfig init method
        init default config of fittable model wrapper
        """
        self.base_config = {
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
            "fit_results": {
                "fit_name": None,
                "running_epoch": None,
                "train_results": None,
                "valid_results": None,
                "used_device": None,
                "num_processes": None,
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

    def __str__(self) -> str:
        """__str__
        default string representation of config

        Returns:
            str: default config as a string
        """
        res = "base_config: \n\n"
        res += str(self.base_config)
        res += str("\n")
        return str(res)
    

class FittableModelAvailables:
    """FittableModelAvailables class for fittable model wrapper

    1. You can insert in optimizers dict new optimizer by
    FittableModelAvailables.optimizers["your_new_cool_optimizer"] = your_new_cool_optimizer
    Default optimizer is Adam torch.optim.Adam

    2. You can insert in schedulers dict new scheduler by
    FittableModelAvailables.schedulers["your_new_cool_scheduler"] = your_new_cool_scheduler
    No default sceduler available

    3. You can insert in metrics dict ner metric by
    FittableModelAvailables.metrics["your_new_cool_metric"] = [your_new_cool_metric, your_new_cool_metric_direction]
    Where your_new_cool_metric_direction is direction for using in overfitting detector for understanding overfitting direction
    No default metrics available
    """

    optimizers = {"Adam": torch.optim.Adam}
    schedulers = {}
    metrics = {}