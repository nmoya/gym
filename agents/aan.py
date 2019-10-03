import numpy as np

w = np.array([0, 0])
b = 0
x0 = np.array([-1, 1])
x1 = np.array([0, -1])
x2 = np.array([10, 1])
y = np.array([1, -1, 1])
def update_b(label_y):
    return b + label_y

def update_w(x, y):
    return w + (x * y)

def activation(value):
    return 1 if value > 0 else 0

def classify(x, w, b):
    return 1 if activation(np.dot(x, w) + b) == 1 else -1

# print(classify(x1, w, b))
for i, xi in enumerate([x0, x1, x2, x0, x1, x2, x0, x1, x2, x0, x1, x2]):
    classification = classify(xi, w, b)
    print ("x[" + str((i%3)+1) + "]")
    if classification == y[i % 3]:
        print ("Correct: ", i % 3)
    else:
        w = update_w(xi, y[i % 3])
        b = update_b(y[i % 3])
        print("Wrong: ", w, b)

print(w)
print(b)
