import yaml
from config.config import Config
from scripts.main import Base_Model

def run_experiments():
    base_config = Config()
    
    # Experiment with different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    for lr in learning_rates:
        config = base_config
        config.training_params['learning_rate'] = lr
        config.save_config(f'E:/Deep/Assignment 1/config/experiment_lr_{lr}.yaml')
        
        # Run experiment
        loaded_config = Config.load_config(f'E:/Deep/Assignment 1/config/experiment_lr_{lr}.yaml')
        model, losses, cv_losses, acc_train, acc_cv = Base_Model(loaded_config)
        
        # Save results
        results = {
            'hyperparameters': loaded_config,
            'performance': {
                'final_train_loss': losses[-1],
                'final_cv_loss': cv_losses[-1],
                'best_train_acc': max(acc_train),
                'best_cv_acc': max(acc_cv)
            }
        }
        
        with open(f'E:/Deep/Assignment 1/results/experiment_lr_{lr}_results.yaml', 'w') as f:
            yaml.dump(results, f, default_flow_style=False) 