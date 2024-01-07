from sklearn import datasets
import numpy as np

iris: dict = datasets.load_iris()

x_training_set = np.array(iris.get("data"))
y = np.array(iris.get("target"))
y_training_set = np.where(y<1, y, 1)

np.random.seed(1) # repeatbility

w = np.random.rand(1,4) * 0.01 # [w1,w2,w3]
bias = 0

def split_train_test(set):
    limit = int(len(set)*0.8)
    return {
        "train": set[:limit],
        "test": set[limit:]
    }

x_training_set_splitted = split_train_test(x_training_set)
y_training_set_splitted = split_train_test(y_training_set)

learning_rate = 0.01

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute(input):
    z = np.dot(w, input.T) + bias
    return sigmoid(z)

def loss(predictions, ground_truth, length_x_set):
    # binary cross entropy
    return np.sum(ground_truth*np.log(predictions) + (1-ground_truth)*np.log(1-predictions)) * -1/length_x_set

def gradient_descent(a):
    dw = np.matmul((a-y_training_set_splitted.get("train")), x_training_set_splitted.get("train"))
    return (dw * 1/len(x_training_set_splitted.get("train")))

def adjust_weights(updated_w):
    global w
    w -= updated_w*learning_rate

epochs = 999

for i in range(epochs):
    print("- EPOCH NUMBER ", i)
    predictions = compute(x_training_set_splitted.get("train"))
    print(predictions)
    cost = loss(predictions, y_training_set_splitted.get("train"), len(x_training_set_splitted.get("train")))
    print(" - loss: ", cost)
    new_w = gradient_descent(predictions)
    adjust_weights(new_w)

print("TEST SET PREDICTIONS: ")
print("Ground truth: ", y_training_set_splitted.get("test"))
predictions = compute(x_training_set_splitted.get("test"))
print(predictions)
cost = loss(predictions, y_training_set_splitted.get("test"), len(x_training_set_splitted.get("test")))
print(" - loss: ", cost)
