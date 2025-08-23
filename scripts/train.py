import numpy as np
from .evaluate import evaluate_loss

def train(model, x_train: np.ndarray, y_train: np.ndarray, x_cv: np.ndarray, y_cv: np.ndarray, epochs: int, batch_size: int, early_stopping: bool = False, patience: int = 10):
        losses = []
        cv_losses = []
        accs_train = []
        accs_cv = []
        best_CV_loss = float('inf')
        best_model = None
        patience_counter = patience
        for epoch in range(epochs):
            loss = 0.0
            acc_train = 0
            total_train = 0
            model.train = True
            for i in range(0, len(x_train), batch_size):
                batch_x = x_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                output = model.forward(batch_x)
                total_train += len(batch_y)
                predicted = np.argmax(output, axis=1)
                acc_train += np.sum(predicted == batch_y)
                loss += model.compute_loss(output, batch_y) * batch_size
                
                model.backward()
                model.step()


            cv_loss = 0.0
            acc_cv = 0
            total_cv = 0
            model.train = False
            for i in range(0, len(x_cv), batch_size):
                batch_x = x_cv[i:i+batch_size]
                batch_y = y_cv[i:i+batch_size]
                output = model.forward(batch_x)

                total_cv += len(batch_y)
                predicted = np.argmax(output, axis=1)
                acc_cv += np.sum(predicted == batch_y)
                cv_loss += model.compute_loss(output, batch_y) * batch_size

            if early_stopping:
                if cv_loss < best_CV_loss:
                    best_CV_loss = cv_loss
                    best_model = model
                    patience_counter = patience
                else:
                    patience_counter -= 1

            if patience_counter <= 0:
                if best_model is not None:
                    model = best_model
                break

            epoch_train_loss = loss / total_train
            epoch_cv_loss = cv_loss / total_cv
            epoch_train_acc = acc_train / total_train
            epoch_cv_acc = acc_cv / total_cv
            
            print(f"Epoch {epoch + 1} completed")
            losses.append(epoch_train_loss)
            print(f"Train Loss: {losses[-1]}")
            cv_losses.append(epoch_cv_loss)
            print(f"CV Loss: {cv_losses[-1]}")
            accs_train.append(epoch_train_acc)
            print(f"Train Accuracy: {accs_train[-1]}")
            accs_cv.append(epoch_cv_acc)
            print(f"CV Accuracy: {accs_cv[-1]}")


        return losses, cv_losses, accs_train, accs_cv


def train_LE(model, x_train: np.ndarray, y_train: np.ndarray, x_cv: np.ndarray, y_cv: np.ndarray, epochs: int, batch_size: int):
    losses = []
    for epoch in range(epochs):
        loss = 0.0
        total_train = 0
        model.train = True
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            output = model.forward(batch_x)

            total_train += len(batch_y)
            loss += model.compute_loss(output, batch_y, False) * batch_size
            model.backward()
            model.step()

        epoch_train_loss = loss / total_train
        
        print(f"Epoch {epoch + 1} completed")
        print(f"Train Loss: {epoch_train_loss}")
        losses.append(epoch_train_loss)


    return losses
    

