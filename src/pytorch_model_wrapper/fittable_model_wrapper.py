import os
import yaml
import copy
from time import time
from tqdm import tqdm
from glob import glob
from typing import Optional, Any, Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

import mlflow
import boto3

from pytorch_model_wrapper.fittable_model_wrapper_commons import FittableModelBaseConfig, FittableModelAvailables
from pytorch_model_wrapper.fittable_model_wrapper_utils import (
    FittableModelStages, 
    FittableModelMonitor, 
    FittableModelLogger, 
    FittableModelMetric, 
    FittableModelHistory,
    FittableModelConfigVSprovidedDataError
)


class FittableModelWrapper:
    def __init__(
        self, model: torch.nn.Module, fittable_model_config: Optional[dict] = None
    ) -> None:
        """FittableModelWrapper init method

        Args:
            model (torch.nn.Module): pytorch model for wrapping
            fittable_model_config (dict, optional): config for FittableModelWrapper in dict format. Defaults to None.
        """
        self.model = model
        self.fittable_model_config = fittable_model_config
        self._init_config()
        self.accelerator = Accelerator(**self.fittable_config["init"]["accelerator"])
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        self.history = FittableModelHistory()
        self.stages = FittableModelStages()
        self.logger = FittableModelLogger(
            self.is_main_process,
            self.fittable_config["init"]["logging_file"],
            self.fittable_config["init"]["log_only_main_process"],
        )
        self.experiment_name = str(self.fittable_config["init"]["experiment_name"])

    def fit(
        self,
        criterion: Any,
        dataset_train: Dataset,
        dataset_valid: Optional[Dataset] = None,
        config_dict: Optional[dict] = None,
    ) -> None:
        """FittableModelWrapper fit method

        FittableModelWrapper requires special dataset format:
        tensor of ids, dict of feature tensors, tensor of targets
        In fit method tensor of ids are not affect on calculations but still needed for providing 

        Args:
            criterion (Any): criterion for training model
            dataset_train (Dataset): train dataset
            dataset_valid (Optional[Dataset], optional): validation dataset if required. Defaults to None.
            config_dict (Optional[dict], optional): dict with possible updatings of your config. Defaults to None.
        """
        self.stage = self.stages.FIT_START
        self.logger.log("INFO", self.stage, self.device)
        self.criterion = criterion
        self._update_config(config_dict)
        self._process_config_fit_method()
        dataloader_train = self._create_dataloader(dataset_train)
        dataloader_valid = self._create_dataloader(dataset_valid)
        self._wrap_accelerator()
        self._fit(dataloader_train, dataloader_valid)
        self._unwrap_accelerator()
        self.stage = self.stages.FIT_END
        self.logger.log("INFO", self.stage, self.device)

    def evaluate(
        self, criterion: Any, dataset: Dataset, config_dict: Optional[dict] = None
    ) -> None:
        """FittableModelWrapper evaluate method
        
        FittableModelWrapper requires special dataset format:
        tensor of ids, dict of feature tensors, tensor of targets
        In evaluate method tensor of ids are not affect on calculations but still needed for providing

        Args:
            criterion (Any): criterion for evaluating model
            dataset (Dataset): dataset for evaluating
            config_dict (Optional[dict], optional): dict with possible updatings of your config. Defaults to None.
        """
        self.stage = self.stages.EVALUATE_START
        self.logger.log("INFO", self.stage, self.device)
        self.criterion = criterion
        self._update_config(config_dict)
        self._process_config_evaluate_method()
        dataloader = self._create_dataloader(dataset)
        self._wrap_accelerator(only_model=True)
        self._evaluate(dataloader)
        self._unwrap_accelerator(only_model=True)
        self.stage = self.stages.EVALUATE_END
        self.logger.log("INFO", self.stage, self.device)

    def predict(self, dataset: Dataset, config_dict: Optional[dict] = None) -> None:
        """FittableModelWrapper predict method

        FittableModelWrapper requires special dataset format:
        tensor of ids, dict of feature tensors, tensor of targets
        In predict method tensor of targets are not affect on calculations but still needed for providing
        
        Args:
            dataset (Dataset): dataset for predicting
            config_dict (Optional[dict], optional): dict with possible updatings of your config. Defaults to None.
        """
        self.stage = self.stages.PREDICT_START
        self.logger.log("INFO", self.stage, self.device)
        self._update_config(config_dict)
        self._process_config_predict_method()
        dataloader = self._create_dataloader(dataset)
        self._wrap_accelerator(only_model=True)
        self._predict(dataloader)
        self._unwrap_accelerator(only_model=True)
        self.stage = self.stages.PREDICT_END
        self.logger.log("INFO", self.stage, self.device)

    def clean_snapshots(self):
        """FittableModelWrapper clean_snapshots method

        Method for cleaning all possible snapshots created during your experiments defined in your config
        
        Raises:
            e: Error removing snapshot file
        """
        self.stage = self.stages.CLEAN_SNAPSHOTS_START
        self.logger.log("INFO", self.stage, self.device)
        if self.accelerator.is_main_process:
            snapshot_files = [
                os.path.join(self.snapshot_dir, f)
                for f in os.listdir(self.snapshot_dir)
                if os.path.isfile(os.path.join(self.snapshot_dir, f))
                and ("snapshot" in f)
            ]
            snapshot_files
            for file_path in snapshot_files:
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        self.stage = self.stages.CLEAN_SNAPSHOTS_ERROR
                        self.logger.log("ERROR", self.stage, file_path, e, self.device)
                        raise e
        self.stage = self.stages.CLEAN_SNAPSHOTS_END
        self.logger.log("INFO", self.stage, self.device)

    def s3_upload(self, config_dict: Optional[dict] = None) -> None:
        """FittableModelWrapper s3_upload method
        
        Method for uploading your results to AWS S3 bucket. 
        1. Check s3_upload part of config
        2. your environ must have variables with names AWS_S3_ACCESS_KEY_ID and AWS_S3_SECRET_ACCESS_KEY with corresponding values

        Args:
            config_dict (Optional[dict], optional): dict with possible updatings of your config. Defaults to None.
        """
        self.stage = self.stages.S3_UPLOAD_START
        self.logger.log("INFO", self.stage, self.device)
        self._update_config(config_dict)
        self._process_config_s3_upload_method()
        self._s3_upload()
        self.stage = self.stages.S3_UPLOAD_END
        self.logger.log("INFO", self.stage, self.device)

    def _batch_predict_step(self, features: Dict[str, Tensor]) -> Tensor:
        return self.model(features)

    def _batch_process_step(
        self, features: Dict[str, Tensor], targets: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = self._batch_predict_step(features)
        loss = self.criterion(outputs, targets)
        return loss, outputs, targets

    def _train_epoch_process(self, dataloader: DataLoader) -> list:
        losses = []
        for batch_idx, batch in enumerate(dataloader):
            with self.accelerator.accumulate(self.model):
                _, features, targets = batch
                loss, _, _ = self._batch_process_step(features, targets)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    if self.clip_grad_value_ is not None:
                        self.accelerator.clip_grad_value_(
                            self.model.parameters(), self.clip_grad_value_
                        )
                    if self.clip_grad_norm_ is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm_
                        )
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
                if self.accelerator.is_main_process and self.monitor_enabled:
                    self.monitor.write(
                        f"epoch_{self.running_epoch}_{self.monitor.LOSS}",
                        loss.item(),
                        batch_idx,
                    )
            if self.check_run_num is not None:
                if batch_idx >= self.check_run_num:
                    break
        with self.accelerator.accumulate(self.model):
            if self.scheduler:
                self.scheduler.step()
        return losses

    def _train_epoch(self, dataloader: DataLoader) -> list:
        self.accelerator.wait_for_everyone()
        self.optimizer.zero_grad()
        self.model.train()
        dataloader_with_bar = tqdm(
            dataloader,
            unit="batch",
            disable=(not self.accelerator.is_local_main_process),
        )
        losses = self._train_epoch_process(dataloader_with_bar)
        self.accelerator.wait_for_everyone()
        return losses

    def _eval_epoch_process(self, dataloader: DataLoader) -> dict:
        dataloader_len = len(dataloader)
        eval_metrics = {}
        outputs_tensors_by_batch = []
        targets_tensors_by_batch = []
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                _, features, targets = batch
                loss, outputs, targets = self._batch_process_step(features, targets)
                gather_loss = self.accelerator.gather(loss).float()
                gather_outputs, gather_targets = self.accelerator.gather_for_metrics(
                    (outputs, targets)
                )
                outputs_tensors_by_batch.append(gather_outputs.cpu())
                targets_tensors_by_batch.append(gather_targets.cpu())
                total_loss += gather_loss.mean().item()
                if self.check_run_num is not None:
                    if batch_idx >= self.check_run_num:
                        break
        self.accelerator.wait_for_everyone()
        outputs_all = torch.cat(outputs_tensors_by_batch, axis=0)
        targets_all = torch.cat(targets_tensors_by_batch, axis=0)
        total_loss /= dataloader_len
        eval_metrics["criterion"] = total_loss
        for metric_name, metric in self.metrics.items():
            metric_res = metric.metric_func(outputs_all, targets_all)
            eval_metrics[metric_name] = metric_res
        return eval_metrics

    def _eval_epoch(self, dataloader: DataLoader) -> dict:
        self.accelerator.wait_for_everyone()
        self.model.eval()
        dataloader_with_bar = tqdm(
            dataloader,
            unit="batch",
            disable=(not self.accelerator.is_local_main_process),
        )
        eval_metrics = self._eval_epoch_process(dataloader_with_bar)
        self.accelerator.wait_for_everyone()
        return eval_metrics

    def _fit_epoch_process(
        self,
        dataloader_train: DataLoader,
        dataloader_valid: Optional[DataLoader] = None,
    ) -> None:
        train_time_start = time()
        self.logger.log("INFO", "TRAIN EPOCH", self.running_epoch, self.device)
        train_losses = self._train_epoch(dataloader_train)
        train_time = time() - train_time_start
        self.logger.log(
            "INFO", "EVAL EPOCH TRAIN DATA", self.running_epoch, self.device
        )
        train_metrics = self._eval_epoch(dataloader_train)
        train_metrics["train_time"] = train_time
        self.logger.log(
            "INFO",
            "EVAL EPOCH TRAIN DATA",
            self.running_epoch,
            "RESULTS",
            train_metrics,
            self.device,
        )
        valid_metrics = {}
        if dataloader_valid:
            self.logger.log(
                "INFO", "EVAL EPOCH VALID DATA", self.running_epoch, self.device
            )
            valid_metrics = self._eval_epoch(dataloader_valid)
            self.logger.log(
                "INFO",
                "EVAL EPOCH VALID DATA",
                self.running_epoch,
                "RESULTS",
                valid_metrics,
                self.device,
            )
        self.history.add_fit_results(
            self.running_epoch, train_losses, train_metrics, valid_metrics
        )
        self.fittable_config["fit_results"]["running_epoch"] = self.running_epoch
        self.fittable_config["fit_results"]["train_results"] = train_metrics
        self.fittable_config["fit_results"]["valid_results"] = valid_metrics
        if self.accelerator.is_main_process and self.monitor_enabled:
            for metric_name, metric_value in train_metrics.items():
                self.monitor.write(
                    f"{self.monitor.METRIC}_train_{metric_name}",
                    metric_value,
                    self.running_epoch,
                )
            for metric_name, metric_value in valid_metrics.items():
                self.monitor.write(
                    f"{self.monitor.METRIC}_valid_{metric_name}",
                    metric_value,
                    self.running_epoch,
                )

    def _fit_results_process(self) -> None:
        self.logger.log(
            "INFO", "EPOCH RESULTS PROCESS ...", self.running_epoch, self.device
        )
        if self.overfitting_detector_enabled:
            self._overfitting_detector_compute()
            if self._overfitting_detector_is_best_model:
                self._save_checkpoint(torchscripted=self.save_torchscripted)
        if self.saving_checkpoint_enabled:
            self._save_fit_epoch_results()
            if not self.overfitting_detector_enabled:
                if ((self.running_epoch + 1) % self.save_checkpoint_every == 0) or (
                    self.running_epoch == self.epochs - 1
                ):
                    self._save_checkpoint(torchscripted=self.save_torchscripted)
        if self.saving_mlflow_checkpoint_enabled:
            if ((self.running_epoch + 1) % self.save_mlflow_checkpoint_every == 0) or (
                self.running_epoch == self.epochs - 1
            ):
                self._save_mlflow_checkpoint()
            if self._overfitting:
                self._save_mlflow_checkpoint(best=True)
        if self.saving_snapshot_enabled:
            if ((self.running_epoch + 1) % self.save_snapshot_every == 0) or (
                self.running_epoch == self.epochs - 1
            ):
                self._save_snapshot()
        if self._overfitting:
            best_checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self.experiment_name}_{self.fit_name}.pt",
            )
            best_checkpoint_torchscripted_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self.experiment_name}_{self.fit_name}_torchscripted.pt",
            )
            self._load_checkpoint(checkpoint_path=best_checkpoint_path)
            if not self.saving_checkpoint_enabled:
                if os.path.isfile(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                if os.path.isfile(best_checkpoint_torchscripted_path):
                    os.remove(best_checkpoint_path)
        self.logger.log(
            "INFO", "EPOCH RESULTS PROCESS ...", self.running_epoch, self.device
        )

    def _fit_epoch(
        self,
        dataloader_train: DataLoader,
        dataloader_valid: Optional[DataLoader] = None,
    ) -> None:
        self.accelerator.wait_for_everyone()
        self._fit_epoch_process(dataloader_train, dataloader_valid)
        self.accelerator.wait_for_everyone()
        self._unwrap_accelerator()
        self._fit_results_process()
        self._wrap_accelerator()
        self.accelerator.wait_for_everyone()

    def _fit(
        self,
        dataloader_train: DataLoader,
        dataloader_valid: Optional[DataLoader] = None,
    ) -> None:
        self.stage = self.stages.FITTING_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            if (dataloader_valid is None) and self.overfitting_detector_enabled:
                raise FittableModelConfigVSprovidedDataError()
            _, self.input_example, _ = next(iter(dataloader_train))
            for epoch in range(self.snapshot_epoch + 1, self.epochs):
                if self._overfitting:
                    break
                self.running_epoch = epoch
                self._fit_epoch(dataloader_train, dataloader_valid)
        except Exception as e:
            self.stage = self.stages.FITTING_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.FITTING_END
        self.logger.log("INFO", self.stage, self.device)

    def _evaluate(self, dataloader: DataLoader) -> None:
        self.stage = self.stages.EVALUATING_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            self.accelerator.wait_for_everyone()
            self.model.eval()
            dataloader_with_bar = tqdm(
                dataloader,
                unit="batch",
                disable=(not self.accelerator.is_local_main_process),
            )
            eval_metrics = self._eval_epoch_process(dataloader_with_bar)
            self.logger.log("INFO", "EVALUATE RESULTS", eval_metrics, self.device)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and self.save_evaluate_results:
                self._save_evaluate_results(eval_metrics)
        except Exception as e:
            self.stage = self.stages.EVALUATING_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.EVALUATING_END
        self.logger.log("INFO", self.stage, self.device)

    def _predict(self, dataloader: DataLoader) -> None:
        self.stage = self.stages.PREDICTING_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            self.accelerator.wait_for_everyone()
            self.model.eval()
            dataloader_with_bar = tqdm(
                dataloader,
                unit="batch",
                disable=(not self.accelerator.is_local_main_process),
            )
            ids_tensors_by_batch = []
            outputs_tensors_by_batch = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_with_bar):
                    ids, features, _ = batch
                    outputs = self._batch_predict_step(features)
                    gather_ids, gather_outputs = self.accelerator.gather_for_metrics(
                        (ids, outputs)
                    )
                    ids_tensors_by_batch.append(gather_ids.cpu())
                    outputs_tensors_by_batch.append(gather_outputs.cpu())
                    if self.check_run_num is not None:
                        if batch_idx >= self.check_run_num:
                            break
            self.accelerator.wait_for_everyone()
            ids_all = torch.cat(ids_tensors_by_batch, axis=0)
            outputs_all = torch.cat(outputs_tensors_by_batch, axis=0)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                torch.save(
                    {"ids": ids_all, "outputs": outputs_all}, self.predict_results_file
                )
        except Exception as e:
            self.stage = self.stages.PREDICTING_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.PREDICTING_END
        self.logger.log("INFO", self.stage, self.device)

    def _s3_upload(self) -> None:
        self.stage = self.stages.S3_UPLOADING_START
        self.logger.log(
            "INFO", self.stage, "UPLOADING FILES", self.s3_files_to_upload, self.device
        )
        try:
            ACCESS_KEY = os.environ["AWS_S3_ACCESS_KEY_ID"]
            SECRET_KEY = os.environ["AWS_S3_SECRET_ACCESS_KEY"]
            if self.accelerator.is_main_process:
                s3 = boto3.client(
                    "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
                )
                for local_file, aws_file in self.s3_files_to_upload.items():
                    s3.upload_file(local_file, self.s3_bucket_name, aws_file)
        except Exception as e:
            self.stage = self.stages.S3_UPLOADING_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.S3_UPLOADING_END
        self.logger.log("INFO", self.stage, self.device)

    def _create_dataloader(self, dataset: Optional[Dataset]) -> Optional[DataLoader]:
        self.stage = self.stages.CREATE_DATALOADER_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            if dataset is not None:
                dataloader = DataLoader(dataset, **self.fittable_config["dataloader"])
                dataloader = self.accelerator.prepare(dataloader)
            else:
                dataloader = None
        except Exception as e:
            self.stage = self.stages.CREATE_DATALOADER_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.CREATE_DATALOADER_END
        self.logger.log("INFO", self.stage, self.device)
        return dataloader

    def _init_config(self) -> None:
        base_config = FittableModelBaseConfig().base_config
        fittable_config = copy.deepcopy(base_config)
        if self.fittable_model_config is not None:
            for config_part in base_config.keys():
                if config_part in self.fittable_model_config.keys():
                    for config_part_content in self.fittable_model_config[
                        config_part
                    ].keys():
                        fittable_config[config_part][
                            config_part_content
                        ] = self.fittable_model_config[config_part][config_part_content]
        self.fittable_config = copy.deepcopy(fittable_config)

    def _update_config(self, config: dict) -> None:
        if config is None:
            return
        for config_part in self.fittable_config.keys():
            if config_part in config.keys():
                for config_part_content in config[config_part].keys():
                    self.fittable_config[config_part][config_part_content] = config[
                        config_part
                    ][config_part_content]

    def _process_common_config(self) -> None:
        self.from_checkpoint = self.fittable_config["common"]["from_checkpoint"]
        self.check_run_num = self.fittable_config["common"]["check_run_num"]

    def _process_optimizer_config(self) -> None:
        optimizer_name = self.fittable_config["optimizer"]["optimizer_name"]
        optimizer_params = self.fittable_config["optimizer"]["optimizer_params"]
        self.optimizer = FittableModelAvailables.optimizers[optimizer_name](
            self.model.parameters(), **optimizer_params
        )

    def _process_scheduler_config(self) -> None:
        scheduler_name = self.fittable_config["scheduler"]["scheduler_name"]
        scheduler_params = self.fittable_config["scheduler"]["scheduler_params"]
        self.scheduler = None
        if scheduler_name:
            self.scheduler = FittableModelAvailables.schedulers[scheduler_name](
                self.optimizer, **scheduler_params
            )

    def _process_monitor_config(self) -> None:
        self.monitor_enabled = self.fittable_config["monitor"]["monitor_enabled"]
        self.monitor_dir = self.fittable_config["monitor"]["monitor_dir"]
        self.monitor = None
        if self.monitor_enabled:
            monitor_exp = f"{self.experiment_name}_{self.fit_name}"
            self.monitor = FittableModelMonitor(self.monitor_dir, monitor_exp)

    def _process_checkpoint_config(self) -> None:
        self.saving_checkpoint_enabled = self.fittable_config["checkpoint"][
            "saving_checkpoint_enabled"
        ]
        self.save_checkpoint_every = self.fittable_config["checkpoint"][
            "save_checkpoint_every"
        ]
        self.save_torchscripted = self.fittable_config["checkpoint"][
            "save_torchscripted"
        ]
        self.checkpoint_dir = self.fittable_config["checkpoint"]["checkpoint_dir"]
        self.saving_without_optimizer_and_scheduler = self.fittable_config[
            "checkpoint"
        ]["saving_without_optimizer_and_scheduler"]
        self.train_history_dir = self.fittable_config["checkpoint"]["train_history_dir"]
        os.makedirs(self.train_history_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _process_snapshot_config(self) -> None:
        self.snapshot_epoch = -1
        self.saving_snapshot_enabled = self.fittable_config["snapshot"][
            "saving_snapshot_enabled"
        ]
        self.save_snapshot_every = self.fittable_config["snapshot"][
            "save_snapshot_every"
        ]
        self.snapshot_dir = self.fittable_config["snapshot"]["snapshot_dir"]
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _process_mlflow_config(self) -> None:
        self.saving_mlflow_checkpoint_enabled = self.fittable_config["mlflow"][
            "saving_mlflow_checkpoint_enabled"
        ]
        self.save_mlflow_checkpoint_every = self.fittable_config["mlflow"][
            "save_mlflow_checkpoint_every"
        ]
        self.save_mlflow_torchscripted = self.fittable_config["mlflow"][
            "save_mlflow_torchscripted"
        ]
        self.mlflow_tracking_server_uri = self.fittable_config["mlflow"][
            "mlflow_tracking_server_uri"
        ]
        self.mlflow_experiment_name = str(
            self.fittable_config["mlflow"]["mlflow_experiment_name"]
        )
        self.mlflow_tags = self.fittable_config["mlflow"]["mlflow_tags"]
        self.mlflow_params = self.fittable_config["mlflow"]["mlflow_params"]
        if self.saving_mlflow_checkpoint_enabled:
            mlflow.set_tracking_uri(self.mlflow_tracking_server_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)

    def _process_fit_config(self) -> None:
        self.fit_name = str(self.fittable_config["fit"]["fit_name"])
        self.epochs = self.fittable_config["fit"]["epochs"]
        self.freezed_params_names = self.fittable_config["fit"]["freezed_params_names"]
        self.clip_grad_value_ = self.fittable_config["fit"]["clip_grad_value_"]
        self.clip_grad_norm_ = self.fittable_config["fit"]["clip_grad_norm_"]
        metrics_names = self.fittable_config["fit"]["metrics"]
        self.metrics = {
            metric_name: FittableModelMetric(metric_name)
            for metric_name in metrics_names
        }
        self.overfitting_detector_enabled = self.fittable_config["fit"][
            "overfitting_detector_enabled"
        ]
        od_metric = self.fittable_config["fit"]["overfitting_detector_metric"]
        self.overfitting_detector_metric = od_metric if od_metric else "criterion"
        self.overfitting_detector_epochs = self.fittable_config["fit"][
            "overfitting_detector_epochs"
        ]
        self._overfitting_detector_is_best_model = True
        self._overfitting_detector_overfitted = False
        self._overfitted_epochs = 0
        self._overfitting_direction = (
            -1
            if self.overfitting_detector_metric == "criterion"
            else FittableModelMetric(self.overfitting_detector_metric).metric_direction
        )
        self._best_epoch = -1
        self._overfitting = (
            self.overfitting_detector_enabled and self._overfitting_detector_overfitted
        )

    def _process_s3_config(self) -> None:
        self.s3_bucket_name = self.fittable_config["s3_uploading"]["bucket_name"]
        self.s3_folder_name = self.fittable_config["s3_uploading"]["folder_name"]
        self.s3_save_model = self.fittable_config["s3_uploading"]["save_model"]
        self.s3_save_model_torchscripted = self.fittable_config["s3_uploading"][
            "save_model_torchscripted"
        ]
        self.s3_save_fit_results = self.fittable_config["s3_uploading"][
            "save_fit_results"
        ]
        self.s3_save_logs = self.fittable_config["s3_uploading"]["save_logs"]
        self.s3_save_monitor_logs = self.fittable_config["s3_uploading"][
            "save_monitor_logs"
        ]
        self.s3_save_evaluate_results = self.fittable_config["s3_uploading"][
            "save_evaluate_results"
        ]
        self.s3_save_predict_results = self.fittable_config["s3_uploading"][
            "save_predict_results"
        ]
        self.fit_name = str(self.fittable_config["fit"]["fit_name"])
        self.checkpoint_dir = self.fittable_config["checkpoint"]["checkpoint_dir"]
        self.train_history_dir = self.fittable_config["checkpoint"]["train_history_dir"]
        self.logging_file = self.fittable_config["init"]["logging_file"]
        self.monitor_dir = self.fittable_config["monitor"]["monitor_dir"]
        self.evaluate_results_file = self.fittable_config["evaluate"][
            "evaluate_results_file"
        ]
        self.predict_results_file = self.fittable_config["predict"][
            "predict_results_file"
        ]

    def _process_evaluate_config(self) -> None:
        self.save_evaluate_results = self.fittable_config["evaluate"][
            "save_evaluate_results"
        ]
        self.evaluate_results_file = self.fittable_config["evaluate"][
            "evaluate_results_file"
        ]
        metrics_names = self.fittable_config["evaluate"]["metrics"]
        self.metrics = {
            metric_name: FittableModelMetric(metric_name)
            for metric_name in metrics_names
        }

    def _process_predict_config(self) -> None:
        self.predict_results_file = self.fittable_config["predict"][
            "predict_results_file"
        ]

    def _process_config_fit_method(self) -> None:
        self.stage = self.stages.PROCESS_CONFIG_FIT_METHOD_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            self._process_common_config()
            self._process_checkpoint_config()
            self._process_snapshot_config()
            self._process_mlflow_config()
            self._process_fit_config()
            self._process_monitor_config()
            if self.from_checkpoint:
                self._load_checkpoint(checkpoint_path=self.from_checkpoint)
            self._load_snapshot()
            for p_name, p in self.model.named_parameters():
                p.requires_grad = True
                if p_name in self.freezed_params_names:
                    p.requires_grad = False
            self._process_optimizer_config()
            self._process_scheduler_config()
            self.model_parametrs_count = sum(p.numel() for p in self.model.parameters())
            self.model_fittable_parametrs_count = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.history.set_fit_name(self.fit_name)

            self.fittable_config["fit_results"]["fit_name"] = self.fit_name
            self.fittable_config["fit_results"]["used_device"] = str(self.device)
            self.fittable_config["fit_results"][
                "num_processes"
            ] = self.accelerator.num_processes
        except Exception as e:
            self.stage = self.stages.PROCESS_CONFIG_FIT_METHOD_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.PROCESS_CONFIG_FIT_METHOD_END
        self.logger.log("INFO", self.stage, self.fittable_config, self.device)

    def _process_config_evaluate_method(self) -> None:
        self.stage = self.stages.PROCESS_CONFIG_EVALUATE_METHOD_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            self._process_common_config()
            self._process_evaluate_config()
            if self.from_checkpoint:
                self._load_checkpoint(
                    checkpoint_path=self.from_checkpoint, only_model=True
                )
        except Exception as e:
            self.stage = self.stages.PROCESS_CONFIG_EVALUATE_METHOD_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.PROCESS_CONFIG_EVALUATE_METHOD_END
        self.logger.log("INFO", self.stage, self.fittable_config, self.device)

    def _process_config_predict_method(self) -> None:
        self.stage = self.stages.PROCESS_CONFIG_PREDICT_METHOD_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            self._process_common_config()
            self._process_predict_config()
            if self.from_checkpoint:
                self._load_checkpoint(
                    checkpoint_path=self.from_checkpoint, only_model=True
                )
        except Exception as e:
            self.stage = self.stages.PROCESS_CONFIG_PREDICT_METHOD_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.PROCESS_CONFIG_PREDICT_METHOD_END
        self.logger.log("INFO", self.stage, self.fittable_config, self.device)

    def _process_config_s3_upload_method(self) -> None:
        self.stage = self.stages.PROCESS_CONFIG_S3_UPLOAD_METHOD_START
        self.logger.log("INFO", self.stage, self.device)
        try:
            self._process_s3_config()
            model_checkpoint_name = (
                f"checkpoint_{self.experiment_name}_{self.fit_name}.pt"
            )
            model_torchscripted_checkpoint_name = (
                f"checkpoint_{self.experiment_name}_{self.fit_name}_torchscripted.pt"
            )
            model_fit_results_name = (
                f"epochs_results_{self.experiment_name}_{self.fit_name}.yaml"
            )
            model_monitor_exp_name = f"{self.experiment_name}_{self.fit_name}"
            model_checkpoint_file = os.path.join(
                self.checkpoint_dir, model_checkpoint_name
            )
            model_torchscripted_checkpoint_file = os.path.join(
                self.checkpoint_dir, model_torchscripted_checkpoint_name
            )
            model_fit_results_file = os.path.join(
                self.train_history_dir, model_fit_results_name
            )
            model_logs_file = self.logging_file
            model_monitor_dir = os.path.join(self.monitor_dir, model_monitor_exp_name)
            self.s3_files_to_upload = {}
            files_to_upload = []
            if self.s3_save_model:
                files_to_upload.append(model_checkpoint_file)
            if self.s3_save_model_torchscripted:
                files_to_upload.append(model_torchscripted_checkpoint_file)
            if self.s3_save_fit_results:
                files_to_upload.append(model_fit_results_file)
            if self.s3_save_logs:
                files_to_upload.append(model_logs_file)
            if self.s3_save_monitor_logs:
                monitor_files = glob(
                    os.path.join(model_monitor_dir, "**", "*.*"), recursive=True
                )
                for f in monitor_files:
                    files_to_upload.append(f)
            if self.s3_save_evaluate_results:
                files_to_upload.append(self.evaluate_results_file)
            if self.s3_save_predict_results:
                files_to_upload.append(self.predict_results_file)
            for f in files_to_upload:
                if os.path.isfile(f):
                    aws_file = f"{self.s3_folder_name}/{os.path.basename(f)}"
                    self.s3_files_to_upload[f] = aws_file
        except Exception as e:
            self.stage = self.stages.PROCESS_CONFIG_S3_UPLOAD_METHOD_ERROR
            self.logger.log("ERROR", self.stage, e, self.device)
            raise e
        self.stage = self.stages.PROCESS_CONFIG_S3_UPLOAD_METHOD_END
        self.logger.log("INFO", self.stage, self.fittable_config, self.device)

    def _wrap_accelerator(self, only_model: bool = False) -> None:
        self.model = self.accelerator.prepare(self.model)
        if only_model:
            return
        self.optimizer = self.accelerator.prepare(self.optimizer)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)

    def _unwrap_accelerator(self, only_model: bool = False) -> None:
        self.model = self.accelerator.unwrap_model(self.model)
        if only_model:
            return
        self.optimizer = self.accelerator.unwrap_model(self.optimizer)
        if self.scheduler:
            self.scheduler = self.accelerator.unwrap_model(self.scheduler)
        self.model.cpu()

    def _save_fit_epoch_results(self) -> None:
        if self.accelerator.is_main_process:
            results = self.history.get_fit_results_by_id(self.fit_name)
            results_file_name = (
                f"epochs_results_{self.experiment_name}_{self.fit_name}.yaml"
            )
            results_path = os.path.join(self.train_history_dir, results_file_name)
            with open(results_path, "w") as outfile:
                yaml.dump(results, outfile, default_flow_style=False)

    def _save_checkpoint(self, torchscripted: bool = False, mlflow: bool = False) -> None:
        if self.accelerator.is_main_process:
            self.model.eval()
            self.model.cpu()
            postfix = "_mlflow" if mlflow else ""
            model_checkpoint_name = (
                f"checkpoint_{self.experiment_name}_{self.fit_name}{postfix}.pt"
            )
            file_full_path = os.path.join(self.checkpoint_dir, model_checkpoint_name)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": None
                    if self.saving_without_optimizer_and_scheduler
                    else self.optimizer.state_dict(),
                    "scheduler_state_dict": None
                    if self.saving_without_optimizer_and_scheduler
                    or (self.scheduler is None)
                    else self.scheduler.state_dict(),
                    "fittable_config": self.fittable_config,
                },
                file_full_path,
            )
            if torchscripted:
                for k, v in self.input_example.items():
                    self.input_example[k] = v.cpu()
                scripted_model = torch.jit.trace(self.model, self.input_example)
                scripted_model_checkpoint_name = f"checkpoint_{self.experiment_name}_{self.fit_name}{postfix}_torchscripted.pt"
                scripted_file_full_path = os.path.join(
                    self.checkpoint_dir, scripted_model_checkpoint_name
                )
                torch.jit.save(scripted_model, scripted_file_full_path)

    def _load_checkpoint(self, checkpoint_path: str, only_model: bool = False) -> None:
        self.accelerator.wait_for_everyone()
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if only_model:
            self.accelerator.wait_for_everyone()
            return
        if checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (checkpoint["scheduler_state_dict"] is not None) and (
            self.scheduler is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.accelerator.wait_for_everyone()

    def _save_mlflow_checkpoint(self, best: bool = False) -> None:
        if self.accelerator.is_main_process:
            postfix = "" if best else "_mlflow"
            model_checkpoint_name = (
                f"checkpoint_{self.experiment_name}_{self.fit_name}{postfix}.pt"
            )
            file_full_path = os.path.join(self.checkpoint_dir, model_checkpoint_name)
            scripted_model_checkpoint_name = f"checkpoint_{self.experiment_name}_{self.fit_name}{postfix}_torchscripted.pt"
            scripted_file_full_path = os.path.join(
                self.checkpoint_dir, scripted_model_checkpoint_name
            )
            if not best:
                self._save_checkpoint(
                    torchscripted=self.save_mlflow_torchscripted, mlflow=True
                )

            with mlflow.start_run():
                mlflow.log_dict(
                    {
                        self.running_epoch: self.history.get_fit_results_by_id(
                            self.fit_name
                        )[self.running_epoch]
                    },
                    "train_results.yaml",
                )
                mlflow.log_dict(self.fittable_config, "fittable_config.yaml")
                if os.path.isfile(file_full_path):
                    mlflow.log_artifact(file_full_path, artifact_path="model")
                    if not best:
                        os.remove(file_full_path)
                if os.path.isfile(scripted_file_full_path):
                    mlflow.log_artifact(
                        scripted_file_full_path, artifact_path="model_scripted"
                    )
                    if not best:
                        os.remove(scripted_file_full_path)

                mlflow.set_tag(
                    "experiment_name",
                    str(self.fittable_config["init"]["experiment_name"]),
                )
                mlflow.set_tag(
                    "optimizer_name",
                    self.fittable_config["optimizer"]["optimizer_name"],
                )
                mlflow.set_tag(
                    "scheduler_name",
                    self.fittable_config["scheduler"]["scheduler_name"],
                )
                mlflow.set_tag(
                    "accelerator_cpu",
                    self.fittable_config["init"]["accelerator"]["cpu"],
                )
                mlflow.set_tag(
                    "accelerator_mixed_precision",
                    self.fittable_config["init"]["accelerator"]["mixed_precision"],
                )
                mlflow.set_tag(
                    "save_mlflow_torchscripted",
                    self.fittable_config["mlflow"]["save_mlflow_torchscripted"],
                )
                mlflow.set_tag("fit_name", str(self.fittable_config["fit"]["fit_name"]))
                mlflow.set_tag(
                    "from_checkpoint",
                    str(self.fittable_config["common"]["from_checkpoint"]),
                )
                mlflow.set_tag(
                    "overfitting_detector_enabled",
                    self.fittable_config["fit"]["overfitting_detector_enabled"],
                )
                mlflow.set_tag(
                    "overfitting_detector_metric",
                    self.fittable_config["fit"]["overfitting_detector_metric"],
                )
                mlflow.set_tag(
                    "used_device", self.fittable_config["fit_results"]["used_device"]
                )
                mlflow.set_tag("best_model", best)
                for tag_name, tag in self.fittable_config["mlflow"][
                    "mlflow_tags"
                ].items():
                    mlflow.set_tag(tag_name, tag)

                mlflow.log_param(
                    "check_run_num", self.fittable_config["common"]["check_run_num"]
                )
                mlflow.log_param(
                    "gradient_accumulation_steps",
                    self.fittable_config["init"]["accelerator"][
                        "gradient_accumulation_steps"
                    ],
                )
                mlflow.log_param(
                    "step_scheduler_with_optimizer",
                    self.fittable_config["init"]["accelerator"][
                        "gradient_accumulation_steps"
                    ],
                )
                mlflow.log_param(
                    "num_processes",
                    self.fittable_config["fit_results"]["num_processes"],
                )
                mlflow.log_param(
                    "dataloader_batch_size",
                    self.fittable_config["dataloader"]["batch_size"],
                )
                mlflow.log_param(
                    "dataloader_shuffle", self.fittable_config["dataloader"]["shuffle"]
                )
                mlflow.log_param(
                    "dataloader_pin_memory",
                    self.fittable_config["dataloader"]["pin_memory"],
                )
                for param_name, param_value in self.fittable_config["optimizer"][
                    "optimizer_params"
                ].items():
                    mlflow.log_param(f"optimizer_{param_name}", param_value)
                for param_name, param_value in self.fittable_config["scheduler"][
                    "scheduler_params"
                ].items():
                    mlflow.log_param(f"scheduler_{param_name}", param_value)
                mlflow.log_param("epochs", self.fittable_config["fit"]["epochs"])
                mlflow.log_param(
                    "clip_grad_value_", self.fittable_config["fit"]["clip_grad_value_"]
                )
                mlflow.log_param(
                    "clip_grad_norm_", self.fittable_config["fit"]["clip_grad_norm_"]
                )
                mlflow.log_param(
                    "overfitting_detector_epochs",
                    self.fittable_config["fit"]["overfitting_detector_epochs"],
                )
                mlflow.log_param("running_epoch", self.running_epoch)
                mlflow.log_param("model_parametrs_count", self.model_parametrs_count)
                mlflow.log_param(
                    "model_fittable_parametrs_count",
                    self.model_fittable_parametrs_count,
                )
                for param_name, param_value in self.fittable_config["mlflow"][
                    "mlflow_params"
                ].items():
                    mlflow.log_param(f"additional_mlflow_{param_name}", param_value)

                for metric_name, metric_value in self.fittable_config["fit_results"][
                    "train_results"
                ].items():
                    mlflow.log_metric(f"train_{metric_name}", metric_value)
                for metric_name, metric_value in self.fittable_config["fit_results"][
                    "valid_results"
                ].items():
                    mlflow.log_metric(f"valid_{metric_name}", metric_value)

    def _save_snapshot(self) -> None:
        if self.accelerator.is_main_process:
            self.model.eval()
            self.model.cpu()
            file_full_path = os.path.join(
                self.snapshot_dir, f"snapshot_{self.fit_name}.pt"
            )
            torch.save(
                {
                    "snapshot_epoch": self.running_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict()
                    if self.scheduler
                    else None,
                    "history_fit": self.history.fit_results,
                    "overfitting_detector_is_best_model": self._overfitting_detector_is_best_model,
                    "overfitting_detector_overfitted": self._overfitting_detector_overfitted,
                    "overfitted_epochs": self._overfitted_epochs,
                    "best_epoch": self._best_epoch,
                    "overfitting": self._overfitting,
                },
                file_full_path,
            )

    def _load_snapshot(self) -> None:
        file_full_path = os.path.join(self.snapshot_dir, f"snapshot_{self.fit_name}.pt")
        if os.path.isfile(file_full_path):
            snapshot = torch.load(file_full_path)
            self.snapshot_epoch = snapshot["snapshot_epoch"]
            self.model.load_state_dict(snapshot["model_state_dict"])
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            scheduler_state_dict = snapshot["scheduler_state_dict"]
            if (scheduler_state_dict is not None) and (self.scheduler is not None):
                self.scheduler.load_state_dict(scheduler_state_dict)
            else:
                self.scheduler = None
            self.history.fit_results = snapshot["history_fit"]
            self._overfitting_detector_is_best_model = snapshot[
                "overfitting_detector_is_best_model"
            ]
            self._overfitting_detector_overfitted = snapshot[
                "overfitting_detector_overfitted"
            ]
            self._overfitted_epochs = snapshot["overfitted_epochs"]
            self._best_epoch = snapshot["best_epoch"]
            self._overfitting = snapshot["overfitting"]
            self.accelerator.wait_for_everyone()

    def _save_evaluate_results(self, results: dict) -> None:
        with open(self.evaluate_results_file, "w") as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

    def _overfitting_detector_compute(self) -> None:
        current_epoch = self.running_epoch
        best_epoch = self._best_epoch
        if best_epoch < 0:
            self._overfitted_epochs = 0
            self._overfitting_detector_is_best_model = True
            self._overfitting_detector_overfitted = False
            self._overfitting = self._overfitting_detector_overfitted
            self._best_epoch = current_epoch
            return
        current_metric = self.history.fit_results[self.fit_name][current_epoch][
            "valid_metrics"
        ][self.overfitting_detector_metric]
        best_metric = self.history.fit_results[self.fit_name][best_epoch][
            "valid_metrics"
        ][self.overfitting_detector_metric]
        metric_diff = current_metric - best_metric
        if self._overfitting_direction * metric_diff >= 0:
            self._overfitted_epochs = 0
            self._overfitting_detector_is_best_model = True
            self._overfitting_detector_overfitted = False
            self._overfitting = self._overfitting_detector_overfitted
            self._best_epoch = current_epoch
        else:
            self._overfitted_epochs += 1
            self._overfitting_detector_is_best_model = False
            self._overfitting_detector_overfitted = False
            self._overfitting = self._overfitting_detector_overfitted
            if self._overfitted_epochs >= self.overfitting_detector_epochs:
                self._overfitting_detector_overfitted = True
                self._overfitting = self._overfitting_detector_overfitted
                self.logger.log(
                    "INFO",
                    "OVERFITTING DETECTOR OVERFITTED",
                    self.overfitting_detector_metric,
                    best_metric,
                    "Epoch",
                    best_epoch,
                )
