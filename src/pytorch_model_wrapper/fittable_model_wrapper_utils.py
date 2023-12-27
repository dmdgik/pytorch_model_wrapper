import os
from typing import Optional, Any, Union
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_wrapper.fittable_model_wrapper_commons import FittableModelAvailables


class FittableModelStages:
    """FittableModelStages

    STAGES of FittableModelWrapper for logging
    """

    FIT_START = "FIT_START"
    FIT_END = "FIT_END"
    EVALUATE_START = "EVALUATE_START"
    EVALUATE_END = "EVALUATE_END"
    PREDICT_START = "PREDICT_START"
    PREDICT_END = "PREDICT_END"
    CLEAN_SNAPSHOTS_START = "CLEAN_SNAPSHOTS_START"
    CLEAN_SNAPSHOTS_ERROR = "CLEAN_SNAPSHOTS_ERROR"
    CLEAN_SNAPSHOTS_END = "CLEAN_SNAPSHOTS_END"
    S3_UPLOAD_START = "S3_UPLOAD_START"
    S3_UPLOAD_END = "S3_UPLOAD_END"
    FITTING_START = "FITTING_START"
    FITTING_ERROR = "FITTING_ERROR"
    FITTING_END = "FITTING_END"
    EVALUATING_START = "EVALUATING_START"
    EVALUATING_ERROR = "EVALUATING_ERROR"
    EVALUATING_END = "EVALUATING_END"
    PREDICTING_START = "PREDICTING_START"
    PREDICTING_ERROR = "PREDICTING_ERROR"
    PREDICTING_END = "PREDICTING_END"
    S3_UPLOADING_START = "S3_UPLOADING_START"
    S3_UPLOADING_ERROR = "S3_UPLOADING_ERROR"
    S3_UPLOADING_END = "S3_UPLOADING_END"
    CREATE_DATALOADER_START = "CREATE_DATALOADER_START"
    CREATE_DATALOADER_ERROR = "CREATE_DATALOADER_ERROR"
    CREATE_DATALOADER_END = "CREATE_DATALOADER_END"
    PROCESS_CONFIG_FIT_METHOD_START = "PROCESS_CONFIG_FIT_METHOD_START"
    PROCESS_CONFIG_FIT_METHOD_ERROR = "PROCESS_CONFIG_FIT_METHOD_ERROR"
    PROCESS_CONFIG_FIT_METHOD_END = "PROCESS_CONFIG_FIT_METHOD_END"
    PROCESS_CONFIG_EVALUATE_METHOD_START = "PROCESS_CONFIG_EVALUATE_METHOD_START"
    PROCESS_CONFIG_EVALUATE_METHOD_ERROR = "PROCESS_CONFIG_EVALUATE_METHOD_ERROR"
    PROCESS_CONFIG_EVALUATE_METHOD_END = "PROCESS_CONFIG_EVALUATE_METHOD_END"
    PROCESS_CONFIG_PREDICT_METHOD_START = "PROCESS_CONFIG_PREDICT_METHOD_START"
    PROCESS_CONFIG_PREDICT_METHOD_ERROR = "PROCESS_CONFIG_PREDICT_METHOD_ERROR"
    PROCESS_CONFIG_PREDICT_METHOD_END = "PROCESS_CONFIG_PREDICT_METHOD_END"
    PROCESS_CONFIG_S3_UPLOAD_METHOD_START = "PROCESS_CONFIG_S3_UPLOAD_METHOD_START"
    PROCESS_CONFIG_S3_UPLOAD_METHOD_ERROR = "PROCESS_CONFIG_S3_UPLOAD_METHOD_ERROR"
    PROCESS_CONFIG_S3_UPLOAD_METHOD_END = "PROCESS_CONFIG_S3_UPLOAD_METHOD_END"


class FittableModelLogger:
    def __init__(
        self,
        is_main_process: bool,
        logger_file: Optional[str] = None,
        log_only_main_process: bool = True,
    ) -> None:
        """FittableModelLogger init method
        FittableModelLogger is loguru based logger wrapper

        Args:
            is_main_process (bool): is current process main
            logger_file (Optional[str], optional): file for logs storing. Defaults to None.
            log_only_main_process (bool, optional): log only main process. Defaults to True.
        """
        self.logger = logger
        self.is_main_process = is_main_process
        self.log_only_main_process = log_only_main_process
        if logger_file:
            self.logger.add(logger_file, level="INFO")
        self.create_log = self.is_main_process if self.log_only_main_process else True

    def log(self, log_level: str, *log_strings: Any) -> None:
        """FittableModelLogger log method

        Args:
            log_level (str): logging level
            *log_strings: array of elements for logging (will convert to string during logging)
        """
        if self.create_log:
            log_strings = [str(s) for s in log_strings]
            string_to_log = " :: ".join(log_strings)
            self.logger.log(log_level, string_to_log)


class FittableModelMonitor:
    def __init__(self, monitor_dir: str, monitor_exp_name: str) -> None:
        """FittableModelMonitor init method
        FittableModelMonitor is tensorboard base monitor

        Args:
            monitor_dir (str): directory for storing experiments and monitor log files
            monitor_exp_name (str): experiment namme for monitoring
        """
        writer_path = os.path.join(monitor_dir, monitor_exp_name)
        self.writer = SummaryWriter(writer_path)
        self.METRIC = "METRIC"
        self.LOSS = "LOSS"

    def write(
        self, metric_name: str, metric_y: Union[float, int], metric_x: int
    ) -> None:
        """FittableModelMonitor write method
        Wrapper for tensorboard write method

        Args:
            metric_name (str): metric name for logging
            metric_y (Union[float, int]): metric value for logging
            metric_x (int): metric position for logging
        """
        self.writer.add_scalar(metric_name, metric_y, metric_x)


class FittableModelHistory:
    def __init__(self) -> None:
        """FittableModelHistory init method
        class for storing fits results of fittable model wrapper
        """
        self.fit_results = {}
        self.fit_name = None

    def get_fit_results_by_id(self, fit_name: str) -> dict:
        """FittableModelHistory get fit results by fit name method

        Args:
            fit_name (str): fit name

        Returns:
            dict: fit results
        """
        return self.fit_results.get(fit_name, {})

    def get_all_fit_results(self) -> dict:
        """FittableModelHistory get all fits results method

        Returns:
            dict: all fits results dict in format {fit_name: dict_of_results}
        """
        return self.fit_results

    def add_fit_results(
        self, epoch: int, train_losses: list, train_metrics: dict, valid_metrics: dict
    ) -> None:
        """FittableModelHistory add epoch results for current fit name method

        Args:
            epoch (int): epoch num
            train_losses (list): list of train losses
            train_metrics (dict): dict of train metrics in format {metric_name: metric_value}
            valid_metrics (dict): dict of valid metrics in format {metric_name: metric_value}
        """
        res_dict = {}
        res_dict["train_losses"] = train_losses
        res_dict["train_metrics"] = train_metrics
        res_dict["valid_metrics"] = valid_metrics
        self.fit_results[self.fit_name][epoch] = res_dict

    def set_fit_name(self, fit_name: str) -> None:
        """FittableModelHistory set current fit name method

        Args:
            fit_name (str): fit name for setting
        """
        self.fit_name = fit_name
        if not (self.fit_name in list(self.fit_results.keys())):
            self.fit_results[self.fit_name] = {}
            

class FittableModelMetric:
    def __init__(self, metric_name: str) -> None:
        """FittableModelMetric init method

        FittableModelMetric generates object with accessing metric
        function and metric direction from FittableModelAvailables

        Args:
            metric_name (str): metric name in FittableModelAvailables.metrics
        """
        self.metric_func = FittableModelAvailables.metrics[metric_name][0]
        self.metric_direction = FittableModelAvailables.metrics[metric_name][1]


class FittableModelConfigVSprovidedDataError(Exception):
    def __init__(self) -> None:
        """FittableModelConfigVSprovidedDataError init method

        FittableModelConfigVSprovidedDataError is special exception
        for missmatch provided config overfitting detector with no validation data in fit function
        """
        self.message = "Provided config turning on overfitting detector, but no valid dataset provided"
        super().__init__(self.message)