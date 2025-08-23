import pandas as pd
import scipy.io
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def data_loader_MNIST():
    train_data = pd.read_csv('D:\Deep\Assignment 1\data\dataset/mnist_train.csv')
    test_data = pd.read_csv('D:\Deep\Assignment 1\data\dataset/mnist_test.csv')

    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    return train_data, test_data

def data_loader_SVHN():
    train_data = scipy.io.loadmat('D:\Deep\Assignment 1\data\dataset/train_32x32.mat')
    test_data = scipy.io.loadmat('D:\Deep\Assignment 1\data\dataset/test_32x32.mat')

    train_images = train_data['X']
    y_train = train_data['y'].squeeze()
    train_images_gray = np.array([resize(rgb2gray(img), (28, 28)) for img in train_images.transpose((3, 0, 1, 2))])
    train_data = pd.DataFrame(train_images_gray.reshape(train_images_gray.shape[0], -1))

    test_images = test_data['X']
    y_test = test_data['y'].squeeze()
    test_images_gray = np.array([resize(rgb2gray(img), (28, 28)) for img in test_images.transpose((3, 0, 1, 2))])
    test_data = pd.DataFrame(test_images_gray.reshape(test_images_gray.shape[0], -1))

    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    return train_data, y_train, test_data, y_test

def data_loader_LE():
    data = pd.read_csv('D:\Deep\Assignment 1\data\dataset/Life Expectancy Data.csv', header=0)
    return data

def preprocess_data_split_LE(data):
    data['Country'] = data['Country'].astype('category').cat.codes
    data['Status'] = data['Status'].astype('category').cat.codes

    data = data.dropna(subset=['Life expectancy '])

    data[' BMI '] = data[' BMI '].replace(np.nan, data[' BMI '].mean())
    data['Polio'] = data['Polio'].replace(np.nan, data['Polio'].mean())
    data['Diphtheria '] = data['Diphtheria '].replace(np.nan, data['Diphtheria '].mean())
    data[' thinness  1-19 years'] = data[' thinness  1-19 years'].replace(np.nan, data[' thinness  1-19 years'].mean())
    data[' thinness 5-9 years'] = data[' thinness 5-9 years'].replace(np.nan, data[' thinness 5-9 years'].mean())
    data['Alcohol'] = data['Alcohol'].replace(np.nan, data['Alcohol'].mean())
    data['Hepatitis B'] = data['Hepatitis B'].replace(np.nan, data['Hepatitis B'].mean())
    data['Total expenditure'] = data['Total expenditure'].replace(np.nan, data['Total expenditure'].mean())
    data['GDP'] = data['GDP'].replace(np.nan, data['GDP'].mean())
    data['Population'] = data['Population'].replace(np.nan, data['Population'].mean())
    data['Schooling'] = data['Schooling'].replace(np.nan, data['Schooling'].mean())
    data['Income composition of resources'] = data['Income composition of resources'].replace(np.nan, data['Income composition of resources'].mean())

    train_data = data[data['Year'] <= 2010]
    test_data = data[data['Year'] > 2010]

    train_data_min = train_data.min()
    train_data_max = train_data.max()
    train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
    test_data_min = test_data.min()
    test_data_max = test_data.max()
    test_data = (test_data - test_data_min) / (test_data_max - test_data_min)

    X_train = train_data.drop(columns=['Life expectancy '])
    y_train = train_data['Life expectancy ']
    X_test = test_data.drop(columns=['Life expectancy '])
    y_test = test_data['Life expectancy ']

    return X_train, y_train, X_test, y_test
    

def preprocess_data(train_data, test_data):
    train_data.columns = test_data.columns

    if len(train_data.shape) > 2 or len(test_data.shape) > 2:
        raise ValueError("Data has more than 2 dimensions (Not Gray scale and it is probably RGB and has 3 channels)")


    train_number = train_data.iloc[:, 0]
    train_images = train_data.iloc[:, 1:]
    test_number = test_data.iloc[:, 0]
    test_images = test_data.iloc[:, 1:]


    mean = train_images.mean(axis=0)
    std = train_images.std(axis=0)
    std = std.replace(0, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    return train_images.astype(float), train_number.astype(int), test_images.astype(float), test_number.astype(int)

def preprocess_data_SVHN(train_data, y_train, test_data, y_test):
    train_data.columns = test_data.columns

    if len(train_data.shape) > 2 or len(test_data.shape) > 2 or len(y_train.shape) > 2 or len(y_test.shape) > 2:
        raise ValueError("Data has more than 2 dimensions (Not Gray scale and it is probably RGB and has 3 channels)")
    
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    y_train = y_train % 10
    y_test = y_test % 10

    return train_data, y_train, test_data, y_test



def split_data(train_images, train_number, test_images, test_number):
    shuffled_indices = np.random.permutation(len(train_images))
    train_images = train_images.iloc[shuffled_indices]
    train_number = train_number.iloc[shuffled_indices]
    number_of_data = len(train_number)
    number_of_train = int(number_of_data * 0.8)
    X_train = train_images[:number_of_train]
    y_train = train_number[:number_of_train]
    X_CV = train_images[number_of_train:]
    y_CV = train_number[number_of_train:]
    X_test = test_images
    y_test = test_number

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_CV.shape)
    # print(y_CV.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    return X_train, y_train, X_CV, y_CV, X_test, y_test

# data = data_loader_LE()
# X_train, y_train, X_test, y_test = preprocess_data_split_LE(data)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

train_data, test_data = data_loader_MNIST()    
train_images, train_number, test_images, test_number = preprocess_data(train_data, test_data)