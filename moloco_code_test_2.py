import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('regression_data.tsv',sep='\t', header=None, names=['X1','X2','Y'])


def prepare_data(input_df):

    # Remove the 201th data point as outlier
    input_df.drop([200], inplace=True)
    X = data[['X1', 'X2']]

    # Add an extra interaction term (X1*X2)
    X['X3'] = data.X1*data.X2
    X = X.values
    Y = data[['Y']].values

    # Add an extra column for bias term
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    b = np.ones((X.shape[0], 1))
    X = np.concatenate((b, X), axis=1)

    return X, Y


def Train(X, Y):
    X.astype(float)

    # Calculate the OLS estimator b hat
    first = np.dot(X.T, X)
    first.astype(np.float16)
    inverse = np.linalg.inv(first)
    second = np.dot(X.T, Y)

    b = np.dot(inverse, second)
    return b


def predict(X, b):
    return np.dot(X,b)


def r2(y_true, y_pred):

    sse = np.square(y_pred - y_true ).sum()
    sst = np.square(y_true - y_true.mean()).sum()
    return 1 - sse/sst


X, Y = prepare_data(data)
b = Train(X, Y)
train_predict = predict(X, b)
print("R^2:", r2(Y,train_predict))


# create a wiremesh for the plane that the predicted values will lie
xx, yy, zz = np.meshgrid(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
Z = combinedArrays.dot(b[0:3])
Z = Z.reshape(Z.shape[0],)

# graph the original data, predicted data, and wiremesh plane
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y, color='r', label='Actual BP')
ax.scatter(X[:, 0], X[:, 1], train_predict, color='g', label='Predicted BP')
ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()
