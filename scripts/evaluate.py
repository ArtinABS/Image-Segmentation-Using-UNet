import numpy as np

def evaluate_loss(model, X_test, y_test, batch_size: int):
    model.train = False

    loss = 0.0
    total = 0
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        output = model.forward(batch_x)
        loss += model.compute_loss(output, batch_y) * batch_size
        total += batch_size

    return loss / total

def evaluate_accuracy(model, X_test, y_test, batch_size: int):
    model.train = False

    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        output = model.forward(batch_x)
        predicted = np.argmax(output, axis=1)
        correct += np.sum(predicted == batch_y)
        total += batch_size

    return correct / total



