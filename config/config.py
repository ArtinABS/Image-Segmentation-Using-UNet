import yaml
import os

class Config:
    def __init__(self, config_path=None):
        if config_path:
            self.load_config(config_path)
        else:
            self.model_params = {
                'input_dim': 21,
                'hidden_dims': [16],
                'output_dim': 1
            }
            
            self.training_params = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 1000,
                'momentum': 0.9,
                'lambda': 0.1,
                'early_stopping': False,
                'patience': 10
            }

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            self.model_params = config["model_parameters"]
            self.training_params = config["training_parameters"]

    def save_config(self, config_path):
        config = {
            "model_parameters": self.model_params,
            "training_parameters": self.training_params
        }
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            print(f"Configuration saved to {config_path}")


