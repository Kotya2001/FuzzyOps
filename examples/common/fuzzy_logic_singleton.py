import numpy as np
from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_logic import BaseRule, SingletonInference


# generating random parameters for triangular numbers
def generate_params(n, low, high):
    p = np.zeros((n, 3))
    for j in range(n):
        p[j, :] = np.random.uniform(low=low, high=high, size=(3,))
    return p


# Creating a test dataset
x = np.arange(start=0.01, stop=1, step=0.01)
# Values of the target variable
r = np.array([9.919, -6.175, 4.372, -3.680, 2.663, -2.227,
              1.742, -2.789, 11.851, -8.565, 0.938, -0.103])
size = r.shape[0]

# There is 1 feature and 1 target variable in the sample.
X = np.random.choice(x, size=size)
X = np.reshape(X, (size, 1))
data = np.hstack((X, np.reshape(r, (size, 1))))

low, high = np.min(data[:, 0]), np.max(data[:, 0])
# let's create random parameters for the triangular membership function
params = generate_params(size, low, high)
params.sort()

# Building a domain domain for the input variable
x = Domain((np.min(data[:, 0]) - 2, np.max(data[:, 0]), 0.1), name='x')
for i in range(data.shape[0]):
    x.create_number('triangular', *params[i, :].tolist(), name=f"x_{i}")

# we build a Singleton-type rule base (the consequent value is one and it is a clear number)
rul = [
    BaseRule(antecedents=[('x', f'x_{i}')], consequent=data[:, -1][i])
    for i in range(data.shape[0])
]

inference_system = SingletonInference(domains={
    'x': x,
}, rules=rul)

# We submit the input data and get the result
input_data = {'x': 0.66}
result = inference_system.compute(input_data)
print(result)