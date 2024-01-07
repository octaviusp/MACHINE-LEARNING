import numpy as np

# num habitaciones, area cuadrada, distancia al centro
np.random.seed(1) # repetible

# x_training_set = casas, tinene tres features, x1,x2,x3.
x_training_set = np.array([[1,100, 50], [2, 80, 20], [3, 150, 500]]) 
# y_training_set = precio que valen las casas del x_training_set(output)
y_training_set = np.array([15000, 20000, 8000])

# inializo random los parametros iniciales
# esto se hace ya que nadie sabe como tienen que valer para dar lasm ejores predcciones
# si no el machine learning no existiria XD
parametros_iniciales = np.random.rand(3, 1)
# bias es un scalar que se suma a la combinacion lineal, un numero mas que tambien se puede aprender
# pero por fines practicos ni lo usamos aca, tampoco afecta taaanto (establece el treshhold)
bias = 0
# learning_rate, (magnitud del paso que dar en lad ireccion del gradiente)
learning_rate = 0.000005

print("Initial weights: ", parametros_iniciales)
print("Initial bias: ", bias)

# simple funcion para hacer "forward propagation"
# es  Parametros . Input_features + bias
# recordar que es producto punto, toda esa cuenta me da un numero REAL
def computar_precio(input):
    input = input.reshape(3,1)
    result =np.dot(parametros_iniciales.T, input)  + bias
    # np.squeeze aplasta la matriz para que me de un escalar y no [[15]]
    return np.squeeze(result)

def adjust_weights(updated_w):
    # apunto a parametros iniciales global
    global parametros_iniciales
    # actualizo los parametros o pesos, simplmente resto lo actualizado por lo que me dio la derivada
    parametros_iniciales -= updated_w*learning_rate

# MSE : loss function para regresion lineal (varianza) hay que minimizarla
def mean_square_error(predictions):
    return np.sum((y_training_set - predictions)**2) * 1/len(x_training_set)

def gradient_descent(z):
    # al/dw, esto me da las derivadas para todos los pesos ya que usa vectorizacion (single instruction multiple data)
    # se calcula a mano con am1
    dw = -2 * (y_training_set-z)*x_training_set
    # ahora sumo las derivadas para cada ejemplo y divido en 3 para sacar la media
    # esot se hace porque cuando calculas las derivadas de un ejemplo
    # tenes que calcularlas para TODOS los ejemplos
    # si no, se aprende bien a predecir 1 sola casa y no todas
    return (np.sum(dw, axis=1, keepdims=True) * 1/3)

# epochs - iteraciones que se entrena el training_set en el algoritmo
# cuando la loss function no cambie opr mucho significa que llegamos al limite de la mejora
# no vale la pena seguir iterando , estamos gastando recursos
# osea si loss function pasa de 0.0000012 a 0.0000011, no tiene sentido iterar mas
epochs = 100
for i in range(epochs):
    print("- EPOCH NUMBER: ", i)
    predictions = []
    # por cada EPOCH meto todo el training set al algoritmo
    # aplico gradient descent y actualizo los pesos
    for i, house in enumerate(x_training_set):
        prediction =  computar_precio(house)
        predictions.append(prediction)
        print("Prediccion: ", prediction)
        print("Deseado: ", y_training_set[i])
    updated_w = gradient_descent(np.array(predictions))
    adjust_weights(updated_w=updated_w)
    print("- LOSS: ", mean_square_error(predictions) / 1000000)
    print("*"*20)