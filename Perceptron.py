class Perceptron:
    def __init__(self, xb, initWeights):
        assert len(xb) == 2
        assert len(initWeights) > 0
        self.xb = xb[0]
        self.weights = initWeights
        self.weights.append(xb[1])

    def toString(self):
        return "weights= "+str(self.weights)

    def setW(self, i, val):
        self.weights[i] = val

    def Threshold(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def scalar(self, x, w):
        assert len(x) == len(w)
        res = 0
        for i in range(0, len(x)):
            res += x[i] * w[i]
        return res

    def hw(self, x):
        assert len(x) == len(self.weights) - 1  # -1 for b
        input = x.copy()
        input.append(self.xb)
        res = self.scalar(input, self.weights)
        output = self.Threshold(res)
        return output


def train_perceptron(p):
    alpha = 0.1
    training_data = [
        {"input": [2, 0], "output":1},
        {"input": [0, 3], "output":0},
        {"input": [3, 0], "output":0},
        {"input": [1, 1], "output":1},
    ]

    errors = True
    counter = 0
    maxIterations = 100

    while errors and counter < maxIterations:
        errors = False
        counter += 1
        for data in training_data:
            r = p.hw(data["input"])
            if(r != data["output"]):
                errors = True
                #print("correcting with " + str(data))
                for i in range(0, len(p.weights) - 1):
                    p.setW(i, p.weights[i] + alpha *
                           (data["output"] - r) * data["input"][i])
                p.setW(i+1, p.weights[i+1] + alpha * (data["output"] - r) * 1)


# calls
p = Perceptron([1, 0.5], [0, 0])
train_perceptron(p)
print(p.toString())
print(p.hw([2, 0]))
