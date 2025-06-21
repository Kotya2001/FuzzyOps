from fuzzyops.fuzzy_nn import Model, process_csv_data
import torch

# Path to your Iris.csv file in the cpp directory
path = "Iris.csv"

n_features = 2
n_terms = [5, 5]
n_out_vars = 3
lr = 3e-4
batch_size = 8
member_func_type = "gauss"
epochs = 100
verbose = True
device = 'cpu'

X, y = process_csv_data(path=path,
                        target_col="Species",
                        n_features=n_features,
                        use_label_encoder=True,
                        drop_index=True)

model = Model(X, y,
              n_terms, n_out_vars,
              lr,
              batch_size, member_func_type,
              epochs, verbose,
              device=device)

m = model.train()
res = m(torch.Tensor([[5.1, 3.5]]))
print(torch.argmax(res, dim=1))