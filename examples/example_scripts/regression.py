from pytorch_model_wrapper import FittableModelWrapper, FittableModelAvailables
from datasets import RegressionDataset
from models import RegressionModel
from metrics import rmse_score
import torch
import yaml
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="experiment config parser")
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="provide your experiment config yaml file"
    )
    experiment_args = parser.parse_args()
    
    EXPERIMENT_CONFIG_FILE = experiment_args.experiment_config
    with open(EXPERIMENT_CONFIG_FILE, "r") as f:
        experiment_config = yaml.safe_load(f)
        
    experiment_availables = {
        "RegressionDataset" : RegressionDataset,
        "RegressionModel" : RegressionModel,
        "rmse_score" : rmse_score,
        "MSELoss" : torch.nn.MSELoss
    }
    FittableModelAvailables.metrics["RMSE"] = [experiment_availables["rmse_score"], -1]
    
    dataset_class = experiment_config["extra"]["dataset"]["dataset_class"]
    dataset_train_params = experiment_config["extra"]["dataset"]["datasets"][0]
    dataset_valid_params = experiment_config["extra"]["dataset"]["datasets"][1]
    dataset_test_params = experiment_config["extra"]["dataset"]["datasets"][2]
    
    model_class = experiment_config["extra"]["model"]["model_class"]
    model_params = experiment_config["extra"]["model"]["model_params"]
    
    criterion_class = experiment_config["extra"]["criterion"]
    
    dataset_train = experiment_availables[dataset_class](**dataset_train_params)
    dataset_valid = experiment_availables[dataset_class](**dataset_valid_params)
    dataset_test = experiment_availables[dataset_class](**dataset_test_params)
    
    model = experiment_availables[model_class](**model_params)
    
    criterion = experiment_availables[criterion_class]()
    
    additional_fit_config = experiment_config["extra"]["additional_wrapper_config_changes"][0]["change_config"]
    new_predicition_config = experiment_config["extra"]["additional_wrapper_config_changes"][1]["change_config"]
    
    model = FittableModelWrapper(model, experiment_config)
    model.fit(criterion=criterion, dataset_train=dataset_train, dataset_valid=dataset_valid)
    model.clean_snapshots()
    model.evaluate(criterion=criterion, dataset=dataset_test)
    model.predict(dataset=dataset_test)
    model.fit(criterion=criterion, dataset_train=dataset_train, dataset_valid=dataset_valid, config_dict=additional_fit_config)
    model.clean_snapshots()
    model.evaluate(criterion=criterion, dataset=dataset_test)
    model.predict(dataset=dataset_test, config_dict=new_predicition_config)
    
    