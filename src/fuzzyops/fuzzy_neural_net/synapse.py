class FuzzyNNSynapse:

    def __init__(self, weight):
        self.value = 0
        self.error = 0
        self.weight = weight

    def setValue(self, value):
        self.value = value

    def getValue(self):
        # print(type(self.value), type(self.weight))
        return self.value * self.weight

    def setError(self, error):
        self.error = error

    def getError(self):
        return self.error

    def applyError(self):
        self.weight = self.weight + self.error * 0.1