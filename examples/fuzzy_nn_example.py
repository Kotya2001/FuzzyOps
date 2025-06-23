"""
Task:
When developing any devices,
it is necessary to compare the development with competitors and estimate the cost of the product,
having a set of features of already existing technology.

 For example, using data from the website https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
about the characteristics of mobile
phones, it is possible to predict the price range (categories from 0 to 3) based on the input data of the characteristics, which means the category to which the new device will belong (for example, when developing a new mobile phone)

Price categories must be set before the algorithm is trained, so that when it is used, it can get a class label and understand
what a specific cost can be for a new product.

 Thus, the task of classification with a teacher is solved using the fuzzy neural network algorithm (ANFIS)

For training, an object-feature matrix is used, consisting of the first 15 features:

    Battery power;
    Presence/absence of Bluetooth technology (binary indication);
    Processor speed in milliseconds;
    Dual SIM card support (binary feature);
    The number of mega-pixels on the front camera;
    4G technology support (binary feature);
    Memory capacity (GB) (not RAM);
    Screen thickness (cm);
    Phone weight (g);
    Number of processor cores;
    The number of mega pixels on the rear camera;
    Screen resolution in pixels (height);
    Screen resolution in pixels (width);
    The amount of RAM;
    Screen height (cm);
    Screen width (cm);

After training the model, you need to provide a vector of values for each feature (the number is listed above).
The output will be a class label for the price category that the device belongs to.

"""

# (The library is already installed in your project)
from fuzzyops.fuzzy_nn import Model
from sklearn.model_selection import train_test_split

import pandas as pd
import torch

# We load the necessary dat and pre-process it a bit
df = pd.read_csv("train.csv")
Y = df["price_range"]
X = df.drop("price_range", axis=1)

# Let's divide the samples into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
n_features = 1

# Converting the data to torch.Tensor
x = torch.Tensor(X_test.iloc[:, 0: n_features].values)
y = torch.Tensor(Y_test[:].values).unsqueeze(1)

# Let's set the number of terms equal to 2 for each input feature
n_terms = [2 for _ in range(n_features)]
# Setting the number of outputs
n_out_vars1 = 4
# Setting the learning step
lr = 3e-4
# Setting the size of the subsample
batch_size = 64
# Setting the type of membership functions
member_func_type = "gauss"
# Let's set the number of epochs
epochs = 10
# Flag to display information during the learning process
verbose = True
# On which device should the model be trained ('cpu', 'cuda')
device = "cpu" # "cuda" - The training will take place at the GPU

# Creating a model
model = Model(X_train.iloc[:, 0: n_features].values, Y_train[:].values,
              n_terms, n_out_vars1,
              lr,
              batch_size,
              member_func_type,
              epochs,
              verbose,
              device=device)

# training the model
m = model.train()
# If the training took place on a GPU, then in order to predict the model, the data provided to it is also necessary
# transfer to GPU (the trained model and prediction data must be on the same device)

# we use the model by feeding a feature vector as input,
# for example, the first object from the test sample, then we determine the price category
if model.device.type == "cpu":
    res = m(x[0, :].unsqueeze(0))
else:
    res = m(x[0, :].unsqueeze(0).cuda())
print(res.cpu())
print(torch.argmax(res.cpu(), dim=1))
