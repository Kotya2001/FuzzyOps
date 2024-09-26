
class Linear:
    def forward(x):
        return x

    def backward(x):
        return x

class Relu:
    def forward(x):
        return 0 if float(x) <= 0 else x

    def backward(x):
        return x



class FuzzyNNNeuron:
    def __init__(self, neuronType = "linear"):
        self.neuronType = neuronType
        self.intoSynapses = []
        self.outSynapses = []

    def addInto(self, toAdd):
        self.intoSynapses.append(toAdd)

    def addOut(self, toAdd):
        self.outSynapses.append(toAdd)

    def doCalculateForward(self, value):
        if self.neuronType == "linear":
            return Linear.forward(value)
        elif self.neuronType == "relu":
            return Relu.forward(value)

    def doCalculateBackward(self, value):
        if self.neuronType == "linear":
            return Linear.backward(value)
        elif self.neuronType == "relu":
            return Relu.backward(value)

    def forward(self):
        z = 0
        for syn in self.intoSynapses:
            v = syn.getValue()
            try:
                z += v
            except Exception:
                z = v + z
        z = self.doCalculateForward(z)
        for syn in self.outSynapses:
            syn.setValue(z)

    def backward(self):
        z = 0
        for syn in self.outSynapses:
            v = syn.getError()
            try:
                z += v
            except Exception:
                z = v + z
        z = self.doCalculateBackward(z)
        for syn in self.intoSynapses:
            syn.setError(z)
        for syn in self.outSynapses:
            syn.applyError()
