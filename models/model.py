import numpy as np
from data.data_loader import preprocess_data, split_data
from utils.metrics import MSE

class layer:
    def __init__(self, input: np.ndarray):
        raise NotImplementedError
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError
    
    def step(self, learning_rate: float) -> None:
        pass

class linear(layer):
    def __init__(self, input_dim: np.ndarray, output_dim: np.ndarray, momentum: float = 0.0, lambda_: float = 0.0):
        self.weights = np.random.normal(loc=0, scale=0.01, size=(input_dim, output_dim))
        self.bias = np.zeros((1, output_dim))
        self.d_weights = np.zeros((input_dim, output_dim))
        self.d_bias = np.zeros((1, output_dim))
        self.momentum = momentum
        self.lambda_ = lambda_

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, upper_gradian: np.ndarray) -> np.ndarray:
        
        self.d_weights = np.dot(self.input.T, upper_gradian) + self.lambda_ * self.L2_derivative()
        self.d_bias = np.sum(upper_gradian, axis=0, keepdims=True)
        return np.dot(upper_gradian, self.weights.T)
    
    def step(self, learning_rate: float):
        self.d_weights = self.momentum * self.d_weights + learning_rate * self.d_weights
        self.d_bias = self.momentum * self.d_bias + learning_rate * self.d_bias
        self.weights -= self.d_weights
        self.bias -= self.d_bias

    def L2_loss(self):
        return 0.5 * self.lambda_ * np.sum(self.weights ** 2)
    
    def L2_derivative(self):
        return self.lambda_ * self.weights
    
    def L1_loss(self):
        return self.lambda_ * np.sum(np.abs(self.weights))

    def L1_derivative(self):
        return self.lambda_ * np.sign(self.weights)

class relu(layer):

    def __init__(self):
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, upper_gradian: np.ndarray) -> np.ndarray:
        return upper_gradian * (self.input > 0)
    
class softmax(layer):
    def __init__(self):
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x - np.max(x, axis=1, keepdims=True)
        self.output = np.exp(self.input) / np.sum(np.exp(self.input), axis=1, keepdims=True)
        return self.output
    
    def backward(self, upper_gradian: np.ndarray) -> np.ndarray:
        lower_gradian = np.empty_like(self.output)
        
        for i in range(self.output.shape[0]):
            single_output = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            lower_gradian[i] = np.dot(jacobian, upper_gradian[i])
        return lower_gradian

class MLP:
    def __init__(self, layers: list[layer], loss, learning_rate: float, momentum: float = 0.0, lambda_: float = 0.0):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lambda_ = lambda_
        self.train = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self) -> np.ndarray:
        if not self.train:
            return
        error = self.loss.backward()
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def compute_loss(self, input: np.ndarray, target: np.ndarray, need_to_onehot: bool = True) -> float:
        main_loss = self.loss.forward(input, target, need_to_onehot)
        l2_loss = sum(layer.L2_loss() for layer in self.layers if isinstance(layer, linear))
        return main_loss + l2_loss
    
    def step(self):
        if not self.train:
            return
        for layer in self.layers:
            layer.step(self.learning_rate)

    def save_weights(self, path: str, name: str):
        for layer in self.layers:
            if isinstance(layer, linear):
                np.save(f"{path}/{name}_weights_{layer.weights.shape[0]}x{layer.weights.shape[1]}.npy", layer.weights)
                np.save(f"{path}/{name}_bias_{layer.bias.shape[0]}x{layer.bias.shape[1]}.npy", layer.bias)

    def load_weights(self, path: str, name: str):
        for layer in self.layers:
            if isinstance(layer, linear):
                layer.weights = np.load(f"{path}/{name}_weights_{layer.weights.shape[0]}x{layer.weights.shape[1]}.npy")
                layer.bias = np.load(f"{path}/{name}_bias_{layer.bias.shape[0]}x{layer.bias.shape[1]}.npy")


