import pandas as pd
from fuzzyops.fuzzy_nn import Model
from sklearn.preprocessing import LabelEncoder
import torch

# Data for training the model ANFIS
classification_data = pd.read_csv("Iris.csv")
# We take only the first 2 signs (it is possible and more)
n_features = 2
# We determine the number of terms for each feature (The fuzzification layer builds as many fuzzy numbers)
n_terms = [5, 5]
# Set the number of output variables (in our case, we predict 3 classes, so there are 3 output variables)
n_out_vars1 = 3
# The learning rate
lr = 3e-4
# the size of the subsample for training
batch_size = 2
# Type of membership function ('gauss' - Gaussian, 'bell' - generalized bell)
member_func_type = "gauss"
# Number of iterations
epochs = 100
# Flag to display information during the learning process
verbose = True
# On which device to train the model ('cpu', 'cuda')
device = "cpu" # "cuda" - The training will take place at the GPU

# Data
X_class, y_class = classification_data.iloc[:, 1: 1 + n_features].values, \
                             classification_data.iloc[:, -1]

# We encode the target variable, since it is represented by a string type
le = LabelEncoder()
y = le.fit_transform(y_class)

# initializing the model
model = Model(X_class, y,
              n_terms, n_out_vars1,
              lr,
              batch_size,
              member_func_type,
              epochs,
              verbose,
              device=device)

# creating an instance of the class
m = model.train()
# If the training took place on a GPU, then in order to predict the model, the data provided to it is also necessary
# transfer to GPU (The trained model and prediction data must be on the same device)
if model.device.type == "cpu":
    res = m(torch.Tensor([[5.1, 3.5]]))
else:
    res = m(torch.Tensor([[5.1, 3.5]]).cuda())
print(res.cpu())
print(torch.argmax(res.cpu(), dim=1))