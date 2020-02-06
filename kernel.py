import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base64 import b16encode

def rgb2hex(color):
    return f"#{''.join(f'{hex(c)[2:].upper():0>2}' for c in (color,color,color))}"

class Kernel:
    def __init__(self):
        self.e = 0.20
        self.x1 = 0
        self.x2 = 0
        self.x3 = 0
        self.w1 = np.random.normal(0, 0.1)
        self.w2 = np.random.normal(0, 0.1)
        self.w3 = np.random.normal(0, 0.1)
        self.b1 = np.random.normal(0, 0.1)
        self.y1 = 0
        self.y2 = 0
        self.trainCorrect = 0
        self.accuracy = 0

    def predict(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = self.x1 * self.w1 + self.x2 * self. w2 + self.x3 * self.w3 + self.b1
        self.y2 = np.tanh(self.y1)
        return self.y2

    def train(self, x1, x2, x3, A):
        for i, a in enumerate(A):
            self.predict(x1[i], x2[i], x3[i])
            ans = 1 if self.y2 > 0.5 else 0
            if ans == a:
                self.trainCorrect += 1
            self.accuracy = math.floor(self.trainCorrect / (i + 1) * 10000) / 100
            # print(x1[i],x2[i],ans, self.accuracy)
            self.b1 -= self.e * ((self.y2 - a) * (1.0 - self.y2 * self.y2))
            self.w1 -= self.e * ((self.y2 - a) * (1.0 - self.y2 * self.y2) * x1[i])
            self.w2 -= self.e * ((self.y2 - a) * (1.0 - self.y2 * self.y2) * x2[i])
            self.w3 -= self.e * ((self.y2 - a) * (1.0 - self.y2 * self.y2) * x3[i])
            pass
        return

trainSize = 10000
X1 = np.random.randint(0, 2, trainSize)
X2 = np.random.randint(0, 2, trainSize)
X3 = np.array([(X2[i] * X1[i]) for i in range(trainSize)])
A  = np.array([(X2[i] ^ X1[i]) for i in range(trainSize)])

kernal = Kernel()
kernal.train(X1, X2, X3, A)
print("Kerneled XOR Accuracy:", str(kernal.accuracy) + "%")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

examples = 5
show = np.zeros((examples,examples,examples))
for i in range(examples):
    for j in range(examples):
        for k in range(examples):
            show[i][j][k] = kernal.predict(i / examples, j / examples, k / examples)
            show[i][j][k] = (show[i][j][k] + 1) / 2
            ax.scatter(i / examples, j / examples, k / examples,c=rgb2hex((int(show[i][j][k]*225))))
            pass
        pass
    pass

ax.set_axis_off()
plt.show()
