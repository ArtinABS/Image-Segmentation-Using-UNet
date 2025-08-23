import numpy as np
import yaml
import logging
import logging
import yaml
from datetime import datetime
from models.model import MLP, linear, relu, softmax
from utils.metrics import MSE, CrossEntropy
from data.data_loader import split_data, preprocess_data, data_loader_MNIST, data_loader_SVHN, preprocess_data_SVHN, data_loader_LE, preprocess_data_split_LE
from scripts.train import train, train_LE
from utils.visualization import plot, plot_predictions, plot_confusion_matrix, plot_loss
from config.config import Config
from scripts.evaluate import evaluate_loss, evaluate_accuracy

np.random.seed(42)



log_data = {
   "logs": []
}

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    log_data["logs"].append(log_entry)
    logging.info(message)

log_file = "D:/Deep/Assignment 1/config/logging.yaml"

def Base_Model_Mnist(config, loss):
    global log_data
 
    train_data, test_data = data_loader_MNIST()
    
    train_images, train_number, test_images, test_number = preprocess_data(train_data, test_data)
    X_train, y_train, X_CV, y_CV, X_test, y_test = split_data(train_images, train_number, test_images, test_number)

    model = MLP([
        linear(config.model_params['input_dim'], config.model_params['hidden_dims'][0]), 
        relu(), 
        linear(config.model_params['hidden_dims'][0], config.model_params['output_dim'], config.training_params['momentum'], config.training_params['lambda']), 
        softmax()
    ], loss, 
       config.training_params['learning_rate'],
       config.training_params['momentum'],
       config.training_params['lambda'])

    losses, cv_losses, acc_train, acc_cv = train(
        model, X_train, y_train, X_CV, y_CV,
        config.training_params['epochs'],
        config.training_params['batch_size'],
        config.training_params['early_stopping'],
        config.training_params['patience']
    )

    model.save_weights('D:/Deep/Assignment 1/models/saved_models', 'best_model_mnist')

    loss_name = 'MSE'
    log = f"""Training Loss:{losses[-1]:.4f},Training Accuracy:{acc_train[-1]:.4f}|CV Loss:{cv_losses[-1]:.4f},CV Accuracy:{acc_cv[-1]:.4f}|{config.model_params['input_dim']}*{config.model_params['hidden_dims'][0]}*{config.model_params['output_dim']}|Loss={loss_name}|lr={config.training_params['learning_rate']}|Mo={config.training_params['momentum']}|Lam={config.training_params['lambda']}|epoch={config.training_params['epochs']}|BatchSize={config.training_params['batch_size']}"""
    log_message(log)

    with open(log_file, 'a') as f:
        yaml.dump(log_data, f)

    output = model.forward(X_test)
    predicted = np.argmax(output, axis=1)
    acc = np.sum(predicted == y_test)
    acc /= len(X_test)
    plot_confusion_matrix(y_test, predicted)

    return losses, cv_losses, acc_train, acc_cv, acc

# config = Config()
# config.load_config('D:/Deep/Assignment 1/config/best_model_mnist_config.yaml')

# losses, cv_losses, acc_train, acc_cv, acc = Base_Model_Mnist(config, MSE())
# print(acc)

# plot(losses, cv_losses, acc_train, acc_cv)

def Base_Model_SVHN(config):
    train_data, y_train, test_data, y_test = data_loader_SVHN()    
    train_images, train_number, test_images, test_number = preprocess_data_SVHN(train_data, y_train, test_data, y_test)

    model = MLP([
        linear(config.model_params['input_dim'], 
               config.model_params['hidden_dims'][0]), 
        relu(), 
        linear(config.model_params['hidden_dims'][0], 
               config.model_params['output_dim'], 
               config.training_params['lambda']), 
        softmax()
    ], MSE(), 
       config.training_params['learning_rate'],
       config.training_params['momentum'],
       config.training_params['lambda'])
    
    model.load_weights('D:/Deep/Assignment 1/models/saved_models', 'best_model_mnist')
    
    losses, cv_losses, acc_train, acc_cv = train(
        model, train_images, train_number, test_images, test_number,
        config.training_params['epochs'],
        config.training_params['batch_size'],
        config.training_params['early_stopping'],
        config.training_params['patience']
    )

    model.save_weights('D:/Deep/Assignment 1/models/saved_models', 'SVHN_model_mnist_trained')

    loss_name = 'MSE'
    log = f"""Training Loss:{losses[-1]:.4f},Training Accuracy:{acc_train[-1]:.4f}|CV Loss:{cv_losses[-1]:.4f},CV Accuracy:{acc_cv[-1]:.4f}|{config.model_params['input_dim']}*{config.model_params['hidden_dims'][0]}*{config.model_params['output_dim']}|Loss={loss_name}|lr={config.training_params['learning_rate']}|Mo={config.training_params['momentum']}|Lam={config.training_params['lambda']}|epoch={config.training_params['epochs']}|BatchSize={config.training_params['batch_size']}"""
    log_message(log)

    with open(log_file, 'a') as f:
        yaml.dump(log_data, f)

    return losses, cv_losses, acc_train, acc_cv

# config = Config()
# config.load_config('D:/Deep/Assignment 1/config/best_model_mnist_config.yaml')


# losses, cv_losses, acc_train, acc_cv = Base_Model_SVHN(config)
# plot(losses, cv_losses, acc_train, acc_cv)


def LE_Model(config):
    data = data_loader_LE()
    X_train, y_train, X_test, y_test = preprocess_data_split_LE(data)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy().reshape(-1, 1)
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy().reshape(-1, 1)
    

    model = MLP([
        linear(config.model_params['input_dim'], config.model_params['hidden_dims'][0]), 
        relu(), 
        linear(config.model_params['hidden_dims'][0], config.model_params['output_dim'], config.training_params['momentum'], config.training_params['lambda']), 
    ], MSE(), 
       config.training_params['learning_rate'],
       config.training_params['momentum'],
       config.training_params['lambda'])

    losses = train_LE(
        model, X_train, y_train, X_test, y_test,
        config.training_params['epochs'],
        config.training_params['batch_size'],
    )

    loss_name = 'MSE'
    log = f"""Training Loss:{losses[-1]:.4f}|{config.model_params['input_dim']}*{config.model_params['hidden_dims'][0]}*{config.model_params['output_dim']}|Loss={loss_name}|lr={config.training_params['learning_rate']}|Mo={config.training_params['momentum']}|Lam={config.training_params['lambda']}|epoch={config.training_params['epochs']}|BatchSize={config.training_params['batch_size']}"""
    log_message(log)

    return losses, model, X_test, y_test

config = Config('D:/Deep/Assignment 1/config/model_config_LE.yaml')

losses, model, X_test, y_test = LE_Model(config)
model.save_weights('D:/Deep/Assignment 1/models/saved_models', 'LE_model_trained_weights')
plot_loss(losses)
y_pred = model.forward(X_test)
model.train = False
loss = model.compute_loss(X_test, y_test, 0)
print(loss)
# 0.2563456729185145
title = f"LE Model Prediction and Actual"
plot_predictions(y_pred, y_test, title, "Prediction", "Actual")

