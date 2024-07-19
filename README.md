
```
 _______        _______ _____  ______ _    _  _____  __   __ _______ __   _ _______ _______
 |       |      |_____|   |   |_____/  \  /  |     |   \_/   |_____| | \  | |       |______
 |_____  |_____ |     | __|__ |    \_   \/   |_____|    |    |     | |  \_| |_____  |______
```
### Description

Reimplementation of the `Clairvoyance` AutoML method from [Espinoza & Dupont et al. 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008857).  The updated version includes regression support, support for all linear/tree-based models, feature selection through modified `Feature-Engine` classes, and bayesian optimization using `Optuna`.  `Clairvoyance` has built-in (optional) functionality to natively address compositionality of data such as next-generation sequencing counts tables from genomics/transcriptomics.

`Clairvoyance` is currently under active development and API is subject to change.


### Details:
`import clairvoyance as cy`


### Installation

```
# Stable:

# via PyPI
pip install clairvoyance_feature_selection


# Developmental:
pip install git+https://github.com/jolespin/clairvoyance
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

# Add 996 noise features (total = 1000 features) in the same range of values as the original features
number_of_noise_features = 996
vmin = X.values.ravel().min()
vmax = X.values.ravel().max()
X_noise = pd.DataFrame(
    data=np.random.RandomState(0).randint(low=int(vmin*10), high=int(vmax*10), size=(150, number_of_noise_features))/10,
    columns=map(lambda j:"noise_{}".format(j+1), range(number_of_noise_features)),
)

X_iris_with_noise = pd.concat([X, X_noise], axis=1)
X_training, X_testing, y_training, y_testing = train_test_split(X_iris_with_noise, y, stratify=y, random_state=0, test_size=0.3)

# Specify model algorithm and parameter grid
estimator=LogisticRegression(max_iter=1000, solver="liblinear")
param_space={
    "C":["float", 0.0, 1.0],
    "penalty":["categorical", ["l1", "l2"]],
}

# Fit the AutoML model
model = BayesianClairvoyanceClassification(estimator, param_space,  n_iter=4, n_trials=50, feature_selection_method="addition", n_jobs=-1, verbose=0, feature_selection_performance_threshold=0.025)
df_results = model.fit_transform(X_training, y_training, cv=3, optimize_with_training_and_testing=True, X_testing=X_testing, y_testing=y_testing)

[I 2024-07-05 12:14:33,611] A new study created in memory with name: n_iter=1
[I 2024-07-05 12:14:33,680] Trial 0 finished with values: [0.7238095238095238, 0.7333333333333333] and parameters: {'C': 0.417022004702574, 'penalty': 'l1'}. 
[I 2024-07-05 12:14:33,866] Trial 1 finished with values: [0.7238095238095239, 0.7333333333333333] and parameters: {'C': 0.30233257263183977, 'penalty': 'l1'}. 
[I 2024-07-05 12:14:34,060] Trial 2 finished with values: [0.39999999999999997, 0
...
Recursive feature addition: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 170.02it/s]
Synopsis[n_iter=2] Input Features: 6, Selected Features: 1
Initial Training Score: 0.9047619047619048, Feature Selected Training Score: 0.8761904761904762
Initial Testing Score: 0.7777777777777778, Feature Selected Testing Score: 0.9333333333333333
```

##### Example output:
We were able to filter out all the noise features and get just the most informative features but linear models might not be the best for this classification task.


| study_name   | best_hyperparameters                       | best_estimator                                                        | best_trial                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |   number_of_initial_features |   initial_training_score |   initial_testing_score |   number_of_selected_features |   feature_selected_training_score |   feature_selected_testing_score | selected_features                                                               |
|:-------------|:-------------------------------------------|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|-------------------------:|------------------------:|------------------------------:|----------------------------------:|---------------------------------:|:--------------------------------------------------------------------------------|
| n_iter=1     | {'C': 0.0745664572902166, 'penalty': 'l1'} | LogisticRegression(C=0.0745664572902166, max_iter=1000, penalty='l1', | FrozenTrial(number=28, state=TrialState.COMPLETE, values=[0.7904761904761904, 0.7333333333333333], datetime_start=datetime.datetime(2024, 7, 6, 15, 53, 9, 422777), datetime_complete=datetime.datetime(2024, 7, 6, 15, 53, 9, 491422), params={'C': 0.0745664572902166, 'penalty': 'l1'}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'C': FloatDistribution(high=1.0, log=False, low=0.0, step=None), 'penalty': CategoricalDistribution(choices=('l1', 'l2'))}, trial_id=28, value=None)  |                         1000 |                 0.790476 |                0.733333 |                             6 |                          0.904762 |                         0.733333 | ['petal_length', 'noise_25', 'noise_833', 'noise_48', 'noise_653', 'noise_793'] |
| n_iter=2     | {'C': 0.9875411040455084, 'penalty': 'l1'} | LogisticRegression(C=0.9875411040455084, max_iter=1000, penalty='l1', | FrozenTrial(number=11, state=TrialState.COMPLETE, values=[0.9047619047619048, 0.7777777777777778], datetime_start=datetime.datetime(2024, 7, 6, 15, 53, 33, 987822), datetime_complete=datetime.datetime(2024, 7, 6, 15, 53, 34, 12108), params={'C': 0.9875411040455084, 'penalty': 'l1'}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'C': FloatDistribution(high=1.0, log=False, low=0.0, step=None), 'penalty': CategoricalDistribution(choices=('l1', 'l2'))}, trial_id=11, value=None) |                            6 |                 0.904762 |                0.777778 |                             1 |                          0.87619  |                         0.933333 | ['petal_length']                                                                |


##### Let's try it again with a tree-based model:

```
# Specify DecisionTree model algorithm and parameter grid
from sklearn.tree import DecisionTreeClassifier

estimator=DecisionTreeClassifier(random_state=0)
param_space = {
    "min_samples_leaf":["int", 1, 50], 
    "min_samples_split": ["float", 0.0, 0.5], 
    "max_features":["categorical", ["sqrt", "log2", None]],
}

model = BayesianClairvoyanceClassification(estimator, param_space,  n_iter=4, n_trials=10, feature_selection_method="addition", n_jobs=-1, verbose=0, feature_selection_performance_threshold=0.0)
df_results = model.fit_transform(X_training, y_training, cv=3, optimize_with_training_and_testing=True, X_testing=X_testing, y_testing=y_testing)
df_results

[I 2024-07-06 15:48:59,235] A new study created in memory with name: n_iter=1
[I 2024-07-06 15:48:59,313] Trial 0 finished with values: [0.3523809523809524, 0.37777777777777777] and parameters: {'min_samples_leaf': 21, 'min_samples_split': 0.36016224672107905, 'max_features': 'log2'}. 
[I 2024-07-06 15:49:00,204] Trial 1 finished with values: [0.9142857142857143, 0.9555555555555556] and parameters: {'min_samples_leaf': 5, 'min_samples_split': 0.09313010568883545, 'max_features': None}. 
[I 2024-07-06 15:49:00,774] Trial 2 finished with values: [0.3523809523809524, 0.37777777777777777] and parameters: {'min_samples_leaf': 21, 'min_samples_split': 0.34260975019837975, 'max_features': 'log2'}. 
...
/Users/jolespin/miniconda3/envs/soothsayer_env/lib/python3.9/site-packages/clairvoyance/feature_selection.py:632: UserWarning: remove_zero_weighted_features=True and removed 995/1000 features
  warnings.warn("remove_zero_weighted_features=True and removed {}/{} features".format((n_features_initial - n_features_after_zero_removal), n_features_initial))
Recursive feature addition: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 164.94it/s]
Synopsis[n_iter=1] Input Features: 1000, Selected Features: 1
Initial Training Score: 0.9142857142857143, Feature Selected Training Score: 0.9619047619047619
Initial Testing Score: 0.9555555555555556, Feature Selected Testing Score: 0.9555555555555556


/Users/jolespin/miniconda3/envs/soothsayer_env/lib/python3.9/site-packages/clairvoyance/bayesian.py:594: UserWarning: Stopping because < 2 features remain ['petal_width']
  warnings.warn(f"Stopping because < 2 features remain {query_features}")
```

##### Example output:
We were able to get much higher perfomance on both the training and testing sets while identifying the most informative feature(s).

| study_name   | best_hyperparameters                                                                    | best_estimator                                                                | best_trial                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |   number_of_initial_features |   initial_training_score |   initial_testing_score |   number_of_selected_features |   feature_selected_training_score |   feature_selected_testing_score | selected_features   |
|:-------------|:----------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|-------------------------:|------------------------:|------------------------------:|----------------------------------:|---------------------------------:|:--------------------|
| n_iter=1     | {'min_samples_leaf': 5, 'min_samples_split': 0.09313010568883545, 'max_features': None} | DecisionTreeClassifier(min_samples_leaf=5,                                    | FrozenTrial(number=1, state=TrialState.COMPLETE, values=[0.9142857142857143, 0.9555555555555556], datetime_start=datetime.datetime(2024, 7, 6, 15, 49, 0, 127973), datetime_complete=datetime.datetime(2024, 7, 6, 15, 49, 0, 204635), params={'min_samples_leaf': 5, 'min_samples_split': 0.09313010568883545, 'max_features': None}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'min_samples_leaf': IntDistribution(high=50, log=False, low=1, step=1), 'min_samples_split': FloatDistribution(high=0.5, log=False, low=0.0, step=None), 'max_features': CategoricalDistribution(choices=('sqrt', 'log2', None))}, trial_id=1, value=None) |                         1000 |                 0.914286 |                0.955556 |                             1 |                          0.961905 |                         0.955556 | ['petal_width']     |

#### Feature selection based on regression tasks

Alright, let's switch it up and model a regression task instead. We are going to do the controversial boston housing dataset just because it's easy. We are going to use the RMSE scorer from `Scikit-Learn` and increase the number of iterations for the bayesian hyperparamter optimzation.

```python
# Load modules
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from clairvoyance.bayesian import BayesianClairvoyanceRegression
from sklearn.metrics import make_scorer

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
X_training, X_testing, y_training, y_testing = train_test_split(X_normalized, y, random_state=0, test_size=0.1)

# Define the parameter space
estimator = DecisionTreeRegressor(random_state=0)
param_space = {
    "min_samples_leaf":["int", 1, 50], 
    "min_samples_split": ["float", 0.0, 0.5], 
    "max_features":["categorical", ["sqrt", "log2", None]],
}
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Fit the AutoML model
model = BayesianClairvoyanceRegression(estimator, param_space,  n_iter=4, n_trials=10, feature_selection_method="addition", n_jobs=-1, verbose=1, feature_selection_performance_threshold=0.0)
df_results = model.fit_transform(X_training, y_training, cv=5, optimize_with_training_and_testing="auto", X_testing=X_testing, y_testing=y_testing)

I 2024-07-06 01:30:03,567] A new study created in memory with name: n_iter=1
[I 2024-07-06 01:30:03,781] Trial 0 finished with values: [-8.199129905056083, -10.15240690512492] and parameters: {'min_samples_leaf': 21, 'min_samples_split': 0.36016224672107905, 'max_features': 'log2'}. 
[I 2024-07-06 01:30:04,653] Trial 1 finished with values: [-4.971853722495094, -6.666700255530846] and parameters: {'min_samples_leaf': 5, 'min_samples_split': 0.09313010568883545, 'max_features': None}. 
[I 2024-07-06 01:30:05,188] Trial 2 finished with values: [-8.230463026740736, -10.167328393077224] and parameters: {'min_samples_leaf': 21, 'min_samples_split': 0.34260975019837975, 'max_features': 'log2'}. 
...
Recursive feature addition: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 116.99it/s]
Synopsis[n_iter=4] Input Features: 3, Selected Features: 3
Initial Training Score: -4.972940969198907, Feature Selected Training Score: -4.972940969198907
Initial Testing Score: -6.313587662660524, Feature Selected Testing Score: -6.313587662660524

```

#### Example output: 

We successfully removed all the noise features and determined that `RM, LSTAT, CRIM` are the most important features. It's a controversial interpretation so I'm not going there but these results agree with what [other researchers](https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155) have determined as well. 

| study_name | best_hyperparameters                                                                      | best_estimator                                                                                                           | best_trial                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | number_of_initial_features | initial_training_score | initial_testing_score | number_of_selected_features | feature_selected_training_score | feature_selected_testing_score | selected_features                                                                                                                |
|------------|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|------------------------|-----------------------|-----------------------------|---------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| n_iter=1   | {'min_samples_leaf': 5, 'min_samples_split': 0.09313010568883545, 'max_features': None}   | DecisionTreeRegressor(min_samples_leaf=5, min_samples_split=0.09313010568883545,                       random_state=0)   | FrozenTrial(number=1, state=TrialState.COMPLETE, values=[-4.971853722495094, -6.666700255530846], datetime_start=datetime.datetime(2024, 7, 6, 1, 30, 4, 256210), datetime_complete=datetime.datetime(2024, 7, 6, 1, 30, 4, 653385), params={'min_samples_leaf': 5, 'min_samples_split': 0.09313010568883545, 'max_features': None}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'min_samples_leaf': IntDistribution(high=50, log=False, low=1, step=1), 'min_samples_split': FloatDistribution(high=0.5, log=False, low=0.0, step=None), 'max_features': CategoricalDistribution(choices=('sqrt', 'log2', None))}, trial_id=1, value=None)     | 1000                       | -4.971853722495094     | -6.666700255530846    | 12                          | -4.167626439610535              | -6.497959383451274             | ['RM', 'LSTAT', 'CRIM', 'DIS', 'TAX', 'noise_657', 'noise_965', 'noise_711', 'noise_213', 'noise_930', 'noise_253', 'noise_484'] |
| n_iter=2   | {'min_samples_leaf': 30, 'min_samples_split': 0.11300600030211794, 'max_features': None}  | DecisionTreeRegressor(min_samples_leaf=30,                       min_samples_split=0.11300600030211794, random_state=0)  | FrozenTrial(number=5, state=TrialState.COMPLETE, values=[-4.971072001107094, -6.2892657979392474], datetime_start=datetime.datetime(2024, 7, 6, 1, 30, 12, 603770), datetime_complete=datetime.datetime(2024, 7, 6, 1, 30, 12, 619502), params={'min_samples_leaf': 30, 'min_samples_split': 0.11300600030211794, 'max_features': None}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'min_samples_leaf': IntDistribution(high=50, log=False, low=1, step=1), 'min_samples_split': FloatDistribution(high=0.5, log=False, low=0.0, step=None), 'max_features': CategoricalDistribution(choices=('sqrt', 'log2', None))}, trial_id=5, value=None) | 12                         | -4.971072001107094     | -6.2892657979392474   | 4                           | -4.944562598653571              | -6.3774459339786524            | ['RM', 'LSTAT', 'CRIM', 'noise_213']                                                                                             |
| n_iter=3   | {'min_samples_leaf': 45, 'min_samples_split': 0.06279265523191813, 'max_features': None}  | DecisionTreeRegressor(min_samples_leaf=45,                       min_samples_split=0.06279265523191813, random_state=0)  | FrozenTrial(number=1, state=TrialState.COMPLETE, values=[-5.236077512452411, -6.670753984555223], datetime_start=datetime.datetime(2024, 7, 6, 1, 30, 14, 831786), datetime_complete=datetime.datetime(2024, 7, 6, 1, 30, 14, 848240), params={'min_samples_leaf': 45, 'min_samples_split': 0.06279265523191813, 'max_features': None}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'min_samples_leaf': IntDistribution(high=50, log=False, low=1, step=1), 'min_samples_split': FloatDistribution(high=0.5, log=False, low=0.0, step=None), 'max_features': CategoricalDistribution(choices=('sqrt', 'log2', None))}, trial_id=1, value=None)  | 4                          | -5.236077512452411     | -6.670753984555223    | 3                           | -5.236077512452413              | -6.670753984555223             | ['RM', 'LSTAT', 'CRIM']                                                                                                          |
| n_iter=4   | {'min_samples_leaf': 30, 'min_samples_split': 0.004493048833777491, 'max_features': None} | DecisionTreeRegressor(min_samples_leaf=30,                       min_samples_split=0.004493048833777491, random_state=0) | FrozenTrial(number=3, state=TrialState.COMPLETE, values=[-4.972940969198907, -6.313587662660524], datetime_start=datetime.datetime(2024, 7, 6, 1, 30, 19, 160978), datetime_complete=datetime.datetime(2024, 7, 6, 1, 30, 19, 177029), params={'min_samples_leaf': 30, 'min_samples_split': 0.004493048833777491, 'max_features': None}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'min_samples_leaf': IntDistribution(high=50, log=False, low=1, step=1), 'min_samples_split': FloatDistribution(high=0.5, log=False, low=0.0, step=None), 'max_features': CategoricalDistribution(choices=('sqrt', 'log2', None))}, trial_id=3, value=None) | 3                          | -4.972940969198907     | -6.313587662660524    | 3                           | -4.972940969198907              | -6.313587662660524             | ['RM', 'LSTAT', 'CRIM']                                                                                                          |