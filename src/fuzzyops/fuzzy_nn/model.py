from collections import OrderedDict
import itertools
from typing import Union, Callable, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .mf_funcs import make_gauss_mfs, GaussMemberFunc, BellMemberFunc, make_bell_mfs

dtype = torch.float

funcs = Union[GaussMemberFunc, BellMemberFunc]
funcs_type = {"gauss": "gauss", "bell": "bell"}


class _FuzzyVar(torch.nn.Module):
    """
    The class of the layer for fuzzification of input variables

    Attributes:
        mfdefs (torch.nn.ModuleDict): Dictionary of membership functions for fuzzification
        padding (int): The padding value for matrix alignment after fuzzification

    Args:
        mfdefs (List[funcs]): A list of membership functions for the input variable
    """

    def __init__(self, mfdefs: List[funcs]):
        super(_FuzzyVar, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self) -> int:
        """
        Returns the number of terms for each input variable

        Returns:
            int: The number of terms
        """

        return len(self.mfdefs)

    def members(self) -> torch.nn.ModuleDict.items:
        """
        Returns a fuzzy term with its membership function

        Returns:
            torch.nn.ModuleDict.items: Dictionary elements of fuzzy terms and membership functions
        """

        return self.mfdefs.items()

    def pad_to(self, new_size: int) -> None:
        """
        The method sets the padding value to align the matrices after fuzzification

        Args:
            new_size (int): New value for padding
        """

        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x: torch.Tensor):
        """
        Method for fuzzification of transmitted values

        Args:
            x (torch.Tensor): Input values for fuzzification

        Yields:
            Tuple[str, torch.Tensor]: The name of the membership function and its values
        """

        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield mfname, yvals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs fuzzification of the transmitted values and returns the results

        Args:
            x (torch.Tensor): Input values for fuzzification

        Returns:
            torch.Tensor: Fuzzification results, including padding, if necessary
        """

        predictions = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            predictions = torch.cat([predictions, torch.zeros(x.shape[0], self.padding)], dim=1)
        return predictions


class _FuzzyLayer(torch.nn.Module):
    """
    A layer class for combining all fuzzy terms

    Attributes:
        varmfs (torch.nn.ModuleDict): Dictionary of fuzzy variables
        varnames (List[str]): Names of input variables

    Args:
        varmfs (List[_FuzzyVar]): List of fuzzy variables
        varnames (List[str], optional): Variable names (if omitted, x0, x1, etc. are used)
    """

    def __init__(self, varmfs: List[_FuzzyVar], varnames=None):
        super(_FuzzyLayer, self).__init__()
        self.varnames = ['x{}'.format(i) for i in range(len(varmfs))] if not varnames else list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self) -> int:
        """
        A property that returns the number of input variables

        Returns:
            int: Number of input variables
        """

        return len(self.varmfs)

    @property
    def max_mfs(self) -> int:
        """
        A property that returns the maximum number of input terms among all variables

        Returns:
            int: Maximum number of input terms
        """

        return max([var.num_mfs for var in self.varmfs.values()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A method for concatenating fuzzy terms into a single tensor

        Args:
            x (torch.Tensor): Input values to be processed

        Returns:
            torch.Tensor: Concatenated tensor of fuzzy terms

        Raises:
            AssertionError: If the number of input values does not match the expected value
        """

        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class _AntecedentLayer(torch.nn.Module):
    """
    The class of the antecedent layer of fuzzy logic rules

    This class is responsible for creating fuzzy rules using antecedents
    (membership functions), which are determined by input fuzzy variables
    It generates rules as the product of the values of the membership functions for
    the corresponding input signals.

    Attributes:
        mf_indices (torch.Tensor): Indexes of membership functions for generated fuzzy rules

    Args:
        varlist (List[_FuzzyVar]): A list of fuzzy variables, each of which contains its own membership functions
    """

    def __init__(self, varlist: List[_FuzzyVar]):
        super(_AntecedentLayer, self).__init__()
        mf_count = [var.num_mfs for var in varlist]
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))

    def num_rules(self) -> int:
        """
        The method returns the number of fuzzy rules

        Returns:
            int: Number of fuzzy rules
        """

        return len(self.mf_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates the antecedents of the corresponding rule and calculates the degrees of rule fulfillment

        Each rule is determined by the product of the values of the membership functions
        associated with the input signals.

        Args:
            x (torch.Tensor): Input values containing the results of fuzzification of variables,
                                expected dimensions (batch_size, num_mfs, feature_size)

        Returns:
            torch.Tensor: The degree of rule fulfillment for fuzzy rules,
                                and the dimensions (batch_size, num_rules)
        """

        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1)).to(x.device)
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)
        return rules


class _ConsequentLayer(torch.nn.Module):
    """
    The class of the fuzzy logic sequent layer

    This class is responsible for calculating the output values of a fuzzy system
    based on the set rules and input data. It includes
    coefficients (weights) that are used to linearly combine
    inputs to produce totals

    Attributes:
        _coeff (torch.Tensor): Layer parameters representing weights
        for a linear combination of input data

    Args:
        d_in (int): The dimension of the input data
        d_rule (int): Number of fuzzy rules
        d_out (int): The dimension of the output data

    Properties:
        coeff() -> torch.Tensor:
            Returns coefficients (weights) layer
    """

    def __init__(self, d_in: int, d_rule: int, d_out: int):
        super(_ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self) -> torch.Tensor:
        """
        A property that returns the weights of the layer

        Returns:
            torch.Tensor: Current coefficients (weights) layer
        """

        return self.coefficients

    @coeff.setter
    def coeff(self, new_coeff: torch.Tensor) -> None:
        """
        Setter for setting new weights

        Args:
            new_coeff (torch.Tensor): New coefficients for the layer

        Raises:
            AssertionError: If the shape of the new coefficients does not match the shape of the current coefficients
        """

        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates output values based on input data and weights

        The method adds a unit offset to the input data,
        and then performs a matrix multiplication of the weights by the input data
        to obtain the predicted output values

        Args:
            x (torch.Tensor): Input data having dimension (batch_size, d_in)

        Returns:
            torch.Tensor: Output values having dimension (batch_size, d_out)
        """

        x_plus = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)


class _NN(torch.nn.Module):
    """
    A class of fuzzy neural network that combines fuzzy rules and linear models

    This class implements a fuzzy neural network consisting of three main layers:
    1. The fuzzification layer of input variables
    2. A layer of antecedents for forming rules
    3. A consequence layer for calculating output values based on rules

    Attributes:
        outvarnames (List[str]): Names of output variables
        num_in (int): The number of input variables
        num_rules (int): The total number of fuzzy rules
        layer (torch.nn.ModuleDict): A dictionary of network layers, including layers of fuzzification,
            antecedents, and consequences

    Args:
        invardefs (List[Tuple[str, List[funcs]]]): A list of tuples,
            where each tuple consists of the name of the input variable and
            a list of membership functions for that variable
        outvarnames (List[str]): A list of names of output variables

    Properties:
        num_out() -> int:
            Returns the number of output variables
        coeff() -> torch.Tensor:
            Returns coefficients of the impact layer
    """

    def __init__(self, invardefs: List[Tuple[str, List[funcs]]],
                 outvarnames: List[str]):
        super(_NN, self).__init__()
        self.outvarnames = outvarnames
        varnames = [v for v, _ in invardefs]
        mfdefs = [_FuzzyVar(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])

        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', _FuzzyLayer(mfdefs, varnames)),
            ('rules', _AntecedentLayer(mfdefs)),
            ('consequent', _ConsequentLayer(self.num_in, self.num_rules, self.num_out)),
        ]))

    @property
    def num_out(self) -> int:
        """
        Returns the number of output variables

        Returns:
            int: Number of output variables
        """

        return len(self.outvarnames)

    @property
    def coeff(self) -> torch.Tensor:
        """
        Returns coefficients of the sequence layer

        Returns:
            torch.Tensor: Current coefficients of the sequence layer
        """

        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff: torch.Tensor) -> None:
        """
        A setter for setting new coefficients

        Args:
            new_coeff (torch.Tensor): New coefficients for the sequence layer
        """

        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x: torch.Tensor, y_actual: torch.Tensor) -> None:
        """
        A method for learning the weights (coefficients) of the consequence layer

        Args:
            x (torch.Tensor): Input data used for training
            y_actual (torch.Tensor): The actual output data to compare the predictions with
        """

        pass

    def input_variables(self) -> torch.nn.ModuleDict.items:
        """
        Returns fuzzy input variables and their membership functions

        Returns:
            torch.nn.ModuleDict.items: Dictionary elements of fuzzy variables and membership functions
        """

        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self) -> List[str]:
        """
        Returns the names of the output variables

        Returns:
            List[str]: Names of output variables
        """

        return self.outvarnames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation and returns the predicted output values

        The input data is transmitted through the fuzzification layer, then processed
        in the antecedent layer to calculate the degrees of rule fulfillment, and finally
        used in the consequences layer to obtain the final output values

        Args:
            x (torch.Tensor): Input values having dimension (batch_size, num_in)

        Returns:
            torch.Tensor: Predicted output values having dimension (batch_size, num_out)
        """

        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


class Model:
    """
    A class for creating and training a fuzzy logic model

    This class is designed to perform regression and classification tasks using
    fuzzy logic. It accepts input data, defines model parameters, and
    performs preprocessing of the data

    Attributes:
        X (np.ndarray): A matrix of features from a data sample
        Y (np.ndarray): The vector of the target variable from the data sample
        n_input_features (int): Number of input features
        n_terms (List[int]): A list containing the number of terms for each input feature
        n_out_vars (int): Number of output variables
        lr (float): The learning step for optimization
        batch_size (int): The size of the subsample for training
        member_func_type (str): Type of membership function ('gauss' - Gaussian membership function
            'bell' is a generalized bell function)
        device (torch.device): The device on which the model will be executed (for example, "cpu" or "cuda")
        epochs (int): The number of epochs for training the model
        scores (list): A list for saving model metrics
        verbose (bool): The flag for the "detailed" output of information about the learning process
        model (torch.nn.Module): The training model is currently undefined

    Args:
        X (np.ndarray): Input data for the model
        Y (np.ndarray): Target values for the model
        n_terms (list[int]): The number of terms for each input variable
        n_out_vars (int): The number of output variables
        lr (float): The learning step
        batch_size (int): The size of the subsample for training
        member_func_type (str): Type of membership function ('gauss' - Gaussian membership function
            'bell' - generalized bell function)
        epochs (int): The number of epochs for training the model
        verbose (bool): The output detail level (False by default)
        device (str): Computing device ('cpu', 'cuda'), default "cpu"
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 n_terms: list[int], n_out_vars: int, lr: float,
                 batch_size: int, member_func_type: str,
                 epochs: int,
                 verbose: bool = False,
                 device: str = "cpu"):
        self.X = X
        self.Y = Y
        self.n_input_features = X.shape[1]
        self.n_terms = n_terms
        self.n_out_vars = n_out_vars
        self.lr = lr
        self.batch_size = batch_size
        self.member_func_type = member_func_type
        self.device = torch.device(device)
        self.epochs = epochs
        self.scores = []
        self.verbose = verbose
        self.model = None

        print(f"Creating an instance of the class" \
              f"with the following hyperparameters\nNumber of input features: {self.n_input_features}\n" \
              f"Number of terms: {self.n_terms}\nNumber of output variables: {self.n_out_vars}\n" \
              f"The learning step: {self.lr}\nSubsample size: {self.batch_size}\n" \
              f"Type of membership  function: {self.member_func_type}\n" \
              f"The size of the subsample for training: {self.batch_size}\n")

    def __str__(self):
        return f"Creating an instance of the class" \
               f"with the following hyperparameters\nNumber of input features: {self.n_input_features}\n" \
               f"Number of terms: {self.n_terms}\nNumber of output variables: {self.n_out_vars}\n" \
               f"The learning step: {self.lr}\nSubsample size: {self.batch_size}\n" \
               f"Type of membership  function: {self.member_func_type}\n" \
               f"The size of the subsample for training: {self.batch_size}\n"

    def __repr__(self):
        return f"Creating an instance of the class" \
               f"with the following hyperparameters\nNumber of input features: {self.n_input_features}\n" \
               f"Number of terms: {self.n_terms}\nNumber of output variables: {self.n_out_vars}\n" \
               f"The learning step: {self.lr}\nSubsample size: {self.batch_size}\n" \
               f"Type of membership  function: {self.member_func_type}\n" \
               f"The size of the subsample for training: {self.batch_size}\n"

    def __preprocess_data(self) -> DataLoader:
        """
        Preprocessing the data and creating a DataLoader

        Converts input data and target values into tensors,
        encodes output variables for classification
        and creates a DataLoader object to provide data in batches.

        Returns:
            DataLoader: A DataLoader object containing preprocessed data.
        """

        x = torch.Tensor(self.X)
        if self.device:
            x = x.to(self.device)

        y = torch.Tensor(self.Y)
        if self.device:
            y = y.to(self.device)

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def __gauss_func(self, x: torch.Tensor) -> Tuple[List]:
        """
        Generates parameters for Gaussian membership functions based on the input data
        
        Calculates minima, maxima, and ranges for each input variable
        and creates cents and sigma for Gaussian membership functions

        Args:
            x (torch.Tensor): Input data for which membership functions will be created

        Returns:
            Tuple[List]: A list of parameters of input variables and their corresponding membership functions
        """

        input_num = x.shape[1]
        min_values, _ = torch.min(x, dim=0)
        max_values, _ = torch.max(x, dim=0)
        ranges = max_values - min_values
        input_vars = []
        for i in range(input_num):
            sigma = ranges[i] / self.n_terms[i]
            mu_list = torch.linspace(min_values[i], max_values[i], self.n_terms[i]).tolist()
            name = 'x{}'.format(i)
            input_vars.append((name, make_gauss_mfs(sigma, mu_list)))
        out_vars = ['y{}'.format(i) for i in range(self.n_out_vars)]
        return input_vars, out_vars

    def __bell_func(self, x: torch.Tensor) -> Tuple[List]:
        """
        Generates parameters for bell-shaped membership functions based on the input data
        
        Calculates the minima and maxima for each input variable and creates parameters
        for bell-shaped membership functions

        Args:
            x (torch.Tensor): Input data for which membership functions will be created

        Returns:
            Tuple[List]: A tuple containing a list of parameters of input variables and their
                corresponding membership functions
        """

        input_num = x.shape[1]
        min_values, _ = torch.min(x, dim=0)
        max_values, _ = torch.max(x, dim=0)
        input_vars = []
        for i in range(input_num):
            a, b = min_values / self.n_terms[i], max_values / self.n_terms[i]
            c_list = torch.linspace(min_values[i], max_values[i], self.n_terms[i]).tolist()
            name = 'x{}'.format(i)
            input_vars.append((name, make_bell_mfs(a, b, c_list)))
        out_vars = ['y{}'.format(i) for i in range(self.n_out_vars)]
        return input_vars, out_vars

    def __compile(self, x: torch.Tensor) -> _NN:
        """
        Compiles a fuzzy neural network model based on the selected type of membership function
        
        Calls methods to generate membership functions and creates an instance of the model
        `_NN`. Transfers the model to the specified device (CPU or GPU)

        Args:
            x (torch.Tensor): The input data on the basis of which the model will be compiled

        Returns:
            _NN: An instance of a fuzzy neural network
        """

        input_vars, out_vars = self.__gauss_func(x) if self.member_func_type == funcs_type[
            "gauss"] else self.__bell_func(x)
        model = _NN(input_vars, out_vars)
        if self.device:
            model.to(self.device)
        return model

    @staticmethod
    def __class_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculates the value of the loss function for the classification task
        
        Uses cross-entropy to determine the difference between predicted
        and actual class labels

        Args:
            inp (torch.Tensor): The predicted values of the model
            target (torch.Tensor): The actual class labels

        Returns:
            float: The value of the loss function
        """

        return torch.nn.CrossEntropyLoss()(inp, target.squeeze().long())


    @staticmethod
    def __calc_class_score(preds: torch.Tensor, y_actual: torch.Tensor, x: torch.Tensor) -> float:
        """
        Calculates the accuracy of the model for the classification task
        
        Determines the percentage of correct predictions among all input data

        Args:
            preds (torch.Tensor): The predicted values of the model
            y_actual (torch.Tensor): The actual class labels
            x (torch.Tensor): Input values

        Returns:
            float: The percentage of correct predictions
        """

        with torch.no_grad():
            corr = torch.sum(y_actual.squeeze().long() == torch.argmax(preds, dim=1))
            total = len(x)
        return corr * 100 / total

    def __train_loop(self, data: DataLoader, model: _NN,
                     criterion: Callable, calc_score: Callable,
                     optimizer: torch.optim.Adam) -> None:

        """
        The main training cycle of the model
        
        Trains the model on the data, updates the weights, and tracks
        the model's performance during training

        Args:
            data (DataLoader): Loader of training data
            model (_NN): A fuzzy neural network model
            criterion (Callable): The loss function used for training
            calc_score (Callable): A function for evaluating the model
            optimizer (torch.optim.Adam): An optimizer for updating model weights
        """

        score_class = 0

        for t in range(self.epochs):
            for x, y_actual in data:
                y_pred = model(x)
                loss = criterion(y_pred, y_actual)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            x, y_actual = data.dataset.tensors
            y_pred = model(x)

            score =  calc_score(y_pred, y_actual, x)
            
            if score > score_class:
                    self.model = model

            self.scores.append(score)
            if self.epochs < 30 or t % 10 == 0:
                if self.verbose:
                    print(f"epoch: {t}, score: {score}")

    def train(self) -> _NN:
        """
        Starts the learning process of the model

        Performs data preprocessing, model compilation, and
        training cycle execution using the specified criteria and optimizer

        Returns:
            _NN: A trained model of a fuzzy neural network
        """

        train_data = self.__preprocess_data()
        x, y = train_data.dataset.tensors
        model = self.__compile(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = self.__class_criterion
        calc_error = self.__calc_class_score 

        self.__train_loop(train_data, model, criterion, calc_error, optimizer)

        return self.model

    def save_model(self, path: str) -> None:
        """
        Saves the state of the trained model to a file
        Saves the model parameters using the specified path

        Args:
            path (str): The path to the file where the model will be saved

        Raises:
            Exception: If the model has not been trained
        """

        if self.model:
            torch.save(self.model.state_dict(), path)
        else:
            raise Exception("The model is not trained")


def process_csv_data(path: str,
                     target_col: str,
                     n_features: int,
                     use_label_encoder: bool,
                     drop_index: bool,
                     split_size: float = 0.2,
                     use_split: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                                       Tuple[np.ndarray, np.ndarray]]:
    """
    An additional function for data preprocessing with the possibility of dividing the sample into train, test

    Args:
        path (str): The path to the data
        target_col (str): The name of the target column
        n_features (int): The number of input attributes
        use_label_encoder (bool): True - use encoding of input features (if they are specified as a string),
            False - no
        drop_index (bool): True - delete the column with indexes, False - no
        split_size (float): The size of the test subsample depends on the size of the entire dataset
        use_split (bool): Use division into train, test

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]]: Preprocessed data
    """

    df = pd.read_csv(path)
    Y = df[target_col]
    X = df.drop(target_col, axis=1)

    if drop_index:
        X = X.drop(X.columns[0], axis=1)

    if use_split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=split_size)

        new_Y_train = Y_train.values
        new_Y_test = Y_test.values

        new_X_train = X_train.values[:, :n_features]
        new_X_test = X_test.values[:, :n_features]

        if use_label_encoder:
            le = LabelEncoder()
            y_train = le.fit_transform(new_Y_train)
            y_test = le.fit_transform(new_Y_test)
        else:
            y_train = new_Y_train
            y_test = new_Y_test

        return new_X_train, new_X_test, y_train, y_test
    else:
        new_Y = Y.values
        new_X = X.values[:, :n_features]

        if use_label_encoder:
            le = LabelEncoder()
            y = le.fit_transform(new_Y)
        else:
            y = new_Y

        return new_X, y
