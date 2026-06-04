
```
 _______        _______ _____  ______ _    _  _____  __   __ _______ __   _ _______ _______
 |       |      |_____|   |   |_____/  \  /  |     |   \_/   |_____| | \  | |       |______
 |_____  |_____ |     | __|__ |    \_   \/   |_____|    |    |     | |  \_| |_____  |______
```
### Description

Revision of the `Clairvoyance` AutoML method from [Espinoza & Dupont et al. 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008857).  The updated version includes regression support, support for all linear/tree-based models, feature selection through modified `Feature-Engine` classes, and bayesian optimization using `Optuna`.  `Clairvoyance` has built-in (optional) functionality to natively address compositionality of data such as next-generation sequencing counts tables from genomics/transcriptomics.

`Clairvoyance` is currently under active development and API is subject to change.


### Details:
`import clairvoyance as cy`


### Installation

```
pip install clairvoyance_feature_selection
```

### Citation

Espinoza JL, Dupont CL, O’Rourke A, Beyhan S, Morales P, Spoering A, et al. (2021) Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach. PLoS Comput Biol 17(3): e1008857. https://doi.org/10.1371/journal.pcbi.1008857

### Development
*Clairvoyance* is currently under active development and undergoing a complete reimplementation from the ground up from the original publication.  The following includes a list of new features: 

*  Bayesian optimization using `Optuna`
*  Supports any linear or tree-based `Scikit-Learn` compatible estimator
*  Supports any `Scikit-Learn` compatible performance metric
*  Supports regression (in addition to classification as in original implementation)
*  Properly implements transformations for compositional data (e.g., CLR and closure) based on the query features for each iteration
*  Option to remove zero weighted features during model refitting
* [Pending] Visualizations for AutoML

### Usage

#### Feature selection based on classification tasks

##### Let's try using a simple Logistic Regression which can be very powerful for some tasks:

Here's a simple usage case for the iris dataset with 996 noise features (total = 1000 features)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from clairvoyance.bayesian import BayesianClairvoyanceClassification

# Load iris dataset
X, y = load_iris(return_X_y=True, as_frame=True)
X.columns = X.columns.map(lambda j: j.split(" (cm")[0].replace(" ","_"))

# Relabel targets
target_names = load_iris().target_names
y = y.map(lambda i: target_names[i])

# Add 96 noise features (total = 100 features) in the same range of values as the original features
number_of_noise_features = 96
vmin = X.values.ravel().min()
vmax = X.values.ravel().max()
X_noise = pd.DataFrame(
    data=np.random.RandomState(0).randint(low=int(vmin*10), high=int(vmax*10), size=(150, number_of_noise_features))/10,
    columns=map(lambda j:"noise_{}".format(j+1), range(number_of_noise_features)),
)

X_iris_with_noise = pd.concat([X, X_noise], axis=1)
X_training, X_validation, y_training, y_validation = train_test_split(X_iris_with_noise, y, stratify=y, random_state=0, test_size=0.3)

# Specify model algorithm and parameter grid
estimator=LogisticRegression(max_iter=10000, solver="saga")
param_space={
    "C":["float", 0.0, 1.0],
    "l1_ratio": ["float", 0.0, 1.0],
}

# Fit the AutoML model
model = BayesianClairvoyanceClassification(estimator, param_space,  n_iter=3, n_trials=5, feature_selection_method="addition", n_jobs=-1, verbose=0, feature_selection_performance_threshold=0.05)
df_results = model.fit_transform(X_training, y_training, cv=3, X_validation=X_validation, y_validation=y_validation)
# df_results.sort_values("feature_selected_validation_score", ascending=False).iloc[0]
# best_hyperparameters                 {'C': 0.417022004702574, 'l1_ratio': 0.7203244...
# best_estimator                       LogisticRegression(C=0.417022004702574, l1_rat...
# best_trial                           FrozenTrial(number=0, state=<TrialState.COMPLE...
# number_of_initial_features                                                         100
# initial_training_score                                                        0.638095
# initial_validation_score                                                           0.8
# number_of_selected_features                                                          1
# feature_selected_training_score                                                0.92381
# feature_selected_validation_score                                                  1.0
# selected_features                                                       [petal_length]
# Name: n_iter=1, dtype: object
```


##### Let's try it again with a tree-based model:

```python
# Specify DecisionTree model algorithm and parameter grid
from sklearn.tree import DecisionTreeClassifier

estimator=DecisionTreeClassifier(random_state=0)
param_space = {
    "min_samples_leaf":["int", 1, 50], 
    "min_samples_split": ["float", 0.0, 0.5], 
    "max_features":["categorical", ["sqrt", "log2", None]],
}

model = BayesianClairvoyanceClassification(estimator, param_space,  n_iter=3, n_trials=5, feature_selection_method="addition", n_jobs=-1, verbose=0, feature_selection_performance_threshold=0.0)
df_results = model.fit_transform(X_training, y_training, cv=3, X_validation=X_validation, y_validation=y_validation)
# df_results.sort_values("feature_selected_validation_score", ascending=False).iloc[0]
# best_hyperparameters                 {'min_samples_leaf': 5, 'min_samples_split': 0...
# best_estimator                       DecisionTreeClassifier(min_samples_leaf=5,\n  ...
# best_trial                           FrozenTrial(number=1, state=<TrialState.COMPLE...
# number_of_initial_features                                                         100
# initial_training_score                                                         0.92381
# initial_validation_score                                                      0.933333
# number_of_selected_features                                                          2
# feature_selected_training_score                                                0.92381
# feature_selected_validation_score                                             0.955556
# selected_features                                          [petal_length, petal_width]
# Name: n_iter=1, dtype: object
```

#### Feature selection based on regression tasks

Alright, let's switch it up and model a regression task instead. We are going to do the controversial boston housing dataset just because it's easy. We are going to use the RMSE scorer from `Scikit-Learn` and increase the number of iterations for the bayesian hyperparamter optimzation.

```python
# Load modules
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from clairvoyance.bayesian import BayesianClairvoyanceRegression
from sklearn.metrics import make_scorer, root_mean_squared_error

# Load Boston data
# from sklearn.datasets import load_boston; boston = load_boston() # Deprecated
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
X = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
y = pd.Series(target)

# Add some noise features to total 1000 features
number_of_noise_features = 1000 - X.shape[1]
X_noise = pd.DataFrame(np.random.RandomState(0).normal(size=(X.shape[0], number_of_noise_features)),  columns=map(lambda j: f"noise_{j}", range(number_of_noise_features)))
X_boston_with_noise = pd.concat([X, X_noise], axis=1)
X_normalized = X_boston_with_noise - X_boston_with_noise.mean(axis=0).values
X_normalized = X_normalized/X_normalized.std(axis=0).values

# Let's fit the model but leave a held out testing set
X_training, X_validation, y_training, y_validation = train_test_split(X_normalized, y, random_state=0, test_size=0.1)

# Define the parameter space
estimator = DecisionTreeRegressor(random_state=0)
param_space = {
    "min_samples_leaf":["int", 1, 50], 
    "min_samples_split": ["float", 0.0, 0.5], 
    "max_features":["categorical", ["sqrt", "log2", None]],
}
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

# Fit the AutoML model
model = BayesianClairvoyanceRegression(estimator, param_space,  scorer=rmse_scorer, n_iter=3, n_trials=5, feature_selection_method="addition", n_jobs=-1, verbose=1, feature_selection_performance_threshold=0.0)
df_results = model.fit_transform(X_training, y_training, cv=5, X_validation=X_validation, y_validation=y_validation)
df_results.sort_values("feature_selected_validation_score", ascending=False).iloc[0]
# best_hyperparameters                 {'min_samples_leaf': 17, 'min_samples_split': ...
# best_estimator                       DecisionTreeRegressor(max_features='sqrt', min...
# best_trial                           FrozenTrial(number=1, state=<TrialState.COMPLE...
# number_of_initial_features                                                          12
# initial_training_score                                                       -5.678469
# initial_validation_score                                                     -8.055088
# number_of_selected_features                                                          4
# feature_selected_training_score                                              -4.366709
# feature_selected_validation_score                                            -6.196976
# selected_features                                               [LSTAT, RM, CRIM, TAX]
# Name: n_iter=2, dtype: object
```