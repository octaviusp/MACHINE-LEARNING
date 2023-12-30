import math

ALPHA = 0.1 # learning rate

initial_weights = [1.0, 2.0, 3.0] # random starting weights
initial_bias = 0
x_train = [[2.0, 3.0, -1.0], [0, 0, 1], [3.0 , 3.0, 1.0]]
y_train = [1, 0, 0]

# linear activation
def linear_forward(w, b, x):
    outputs = []
    for index, example in enumerate(x):
        outputs.append([])
        z = b
        for i, feature in enumerate(example):
            z += w[i] * example[i]
        outputs[index] = z
    return outputs

linear_output = linear_forward(initial_weights, initial_bias, x_train)
print("linear output: ", linear_output)

def sigmoid(z):
    return 1 / (1 + 2.71**-z)

def sigmoid_forward(w,b,x):
    outputs = []
    for index, example in enumerate(x):
        outputs.append([])
        z = b
        for i, feature in enumerate(example):
            z += w[i] * example[i]
        outputs[index] = sigmoid(z)
    return outputs
  
sigmoid_output = sigmoid_forward(initial_weights, initial_bias, x_train)
print("sigmoid output :", sigmoid_output)
 
print("desired output :", y_train)

def binary_cross_entropy(desired, obtained):
    return desired*math.log(obtained) + (1-desired)*math.log(1-obtained)

def compute_loss(y_train, y_predicted):
    loss = 0
    for i in range(len(y_train)):
        loss += binary_cross_entropy(y_train[i], y_predicted[i])
    return  -1/len(y_train) * loss
    
print("LOSS WITHOUT TRAINING WEIGHTS: ", compute_loss(y_train, sigmoid_output))

# train weights

def train(w, b, learning_rate, x_train, y_train, epochs=10):
    configs = []
    updated_weights = w
    updated_bias = b
    for epoch in range(epochs):
        updated_weights = list(map(lambda weight: weight - ALPHA, updated_weights))
        updated_bias += ALPHA
        new_sigmoid_output = sigmoid_forward(updated_weights, updated_bias, x_train)
        loss = compute_loss(y_train, new_sigmoid_output)
        print(f"\n- LOSS IN EPOCH {epoch}: ", loss)
        configs.append({
            "weights": updated_weights,
            "bias": updated_bias,
            "loss_score": loss,
            "epoch": epoch
        })
    return configs
    
configs = train(initial_weights, initial_bias, ALPHA, x_train, y_train,epochs=50)

print(configs)
def choose_best_config(configs):
    min_loss = configs[0]["loss_score"]
    best_config = configs[0]
    for config in configs:
        if config["loss_score"] < min_loss:
            best_config = config
            min_loss = config["loss_score"]
    return best_config

print("The best config for the machine is: ", choose_best_config(configs))
