import numpy as np

class MSE:

    def __init__(self):
        self.input = None
        self.target = None
        self.output = None

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def forward(self, input: np.ndarray, target: np.ndarray, need_to_onehot : bool = True) -> float:
        self.input = input
        if need_to_onehot:
            self.target = np.eye(input.shape[1])[target.astype(int)]
        else:
            self.target = target
        self.output = np.mean((self.input - self.target) ** 2)
        return self.output

    def backward(self) -> np.ndarray:
        return 2 * (self.input - self.target) / self.input.size
    

class CrossEntropy:
    def __init__(self):
        self.input = None
        self.target = None
        self.output = None

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def forward(self, input: np.ndarray, target: np.ndarray, need_to_onehot : bool = True) -> float:
        self.input = input
        if need_to_onehot:
            self.target = np.eye(10)[target]
        else:
            self.target = target

        eps = 1e-10
        self.input = np.clip(self.input, eps, 1 - eps)

        self.output = -np.sum(self.target * np.log(self.input)) / input.shape[0]
        return self.output  
    
    def backward(self) -> np.ndarray:
        return (self.input - self.target) / self.input.shape[0]
