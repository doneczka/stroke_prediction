# Basic data manipulation and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import List, Set, Dict, Tuple, Optional
from numpy import float64
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from collections import Counter

# Statistical inference
import statsmodels.api as sm
import statsmodels.stats.weightstats as smweight
import statsmodels.stats.proportion as smprop
from scipy.stats import chisquare


# Models
from sklearn.model_selection import (
    train_test_split, 
    cross_validate, 
    GridSearchCV, 
    KFold, 
    StratifiedKFold, 
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier, 
    BaggingClassifier,
)
from sklearn import tree
import xgboost as xg
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from imblearn.under_sampling import RandomUnderSampler
from catboost import Pool, CatBoostClassifier, cv

# Metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    plot_confusion_matrix,

)
from sklearn.model_selection import cross_val_score

# Functions

def create_boxplot(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns boxplot for numerical features of a dataset.
    Boxplots give us a good understanding of how data are spread out in our dataset.
    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a boxplot for.
    Returns: subplots of boxplots for each of the features.
    
    '''
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for i, (ax, curve) in enumerate(zip(axs.flat, list_of_columns)):
        sns.boxplot(y=data[curve], color='darkorange', ax = ax,
                showmeans=True,  meanprops={"marker":"o",
                                            "markerfacecolor":"black", 
                                            "markeredgecolor":"black",
                                            "markersize":"6"},
                                 flierprops={'marker':'o', 
                                             'markeredgecolor':'darkgreen'})

        ax.set_title(list_of_columns[i])
        ax.set_ylabel('') 


def create_histplot(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns histomgrams for numerical features of a dataset.
    Histomgrams give us a good understanding of how data are distributed.
    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a histomgram for.
    Returns: subplots of histomgrams for each of the features.
    
    '''
    fig, axes = plt.subplots(
    1, 3, figsize=(16, 6), gridspec_kw={"hspace": 0.75, "wspace": 0.25})
    for i, ax in enumerate(axes.flatten()):
        sns.histplot(
        data=data, x=data[list_of_columns[i]], ax=ax, kde=True
        )
        ax.ticklabel_format(style="plain")
        ax.set_xlabel('')
        ax.set_title(
        f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
        ax.ticklabel_format(style='sci')

    sns.despine(left=True)
    plt.show()  

def calculate_percentage(data: pd.DataFrame, feature_name: str):
    return (
        data.groupby([feature_name, "stroke"])
        .agg({"stroke": "count"})
        .rename(columns={"stroke": "nr_of_patients"})
        .groupby(level=0)
        .transform(lambda x: x / x.sum() * 100)
        .style.background_gradient(cmap="autumn")
    )


def show_distribution(data, output, column):
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data[data[output] == "No"][column], ax=ax[0], kde=True)
    ax[0].set_title(f"{column} for people who did not have stroke")
    sns.histplot(data[data[output] == "Yes"][column], ax=ax[1], kde=True)
    ax[1].set_title(f"{column} for people who had stroke")
    plt.show()


def calculate_roc_auc(pipelines, X_train, y_train) -> pd.DataFrame:
    model_name = []
    results_mean = []
    results_std = []
    roc_auc = []
    for pipe, model in pipelines:
        kfold = KFold(n_splits=5)
        crossv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring="roc_auc"
        )
        model_name.append(pipe[0:20])
        results_mean.append(crossv_results.mean())
        results_std.append(crossv_results.std())
        roc_auc.append(crossv_results)
    models_comparison = pd.DataFrame(
        {"CV mean": results_mean, "Std": results_std}, index=model_name
    )
    return model_name, roc_auc, models_comparison

def create_countplot(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns countplot for categorical features of a dataset.
    Countplots give us a good understanding of how many instances are represented by specific discrete feature.
    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a countplot for.
    Returns: subplots of countplots for each of the features.
    
    '''
    fig, axes = plt.subplots(1, len(list_of_columns), figsize=(26, 6))
    for i, ax in enumerate(axes.flatten()):
        sns.countplot(data=data, x=data[list_of_columns[i]], ax=ax, order = data[list_of_columns[i]].value_counts().index)
        ax.set_xlabel('')
        ax.set_title(
        f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
    
        sns.despine(left=True)


def create_confusion_matrix(
    dict_of_models: Dict, X_test: np.array, y_test: np.array
) -> plt.figure:
    """Functions that create confusion matrix for machine learning model outcome

    Arg: dict_of_models: dictonary of models with names as keys and models as valus.
        X_test: np.array,
        y_test: np.array - numpy arrays with train/test data
    Return: confusion matric - plt.figure
    """
    f, ax = plt.subplots(1, 6, figsize=(24, 4))
    i = 0
    for key, value in dict_of_models.items():
        y_pred = cross_val_predict(value, X_test, y_test, cv=6)
        sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax[i], annot=True, fmt="2.0f")
        ax[i].set_title(f"Matrix for {key}")
        i += 1
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )


def calculate_roc_auc_models(models, X_train, y_train, classifiers) -> pd.DataFrame:
    results_mean = []
    results_std = []
    roc_auc = []
    for model in models:
        kfold = StratifiedKFold(n_splits=5)
        crossv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring="roc_auc"
        )
        results_mean.append(crossv_results.mean())
        results_std.append(crossv_results.std())
        roc_auc.append(crossv_results)
    models_comparison = pd.DataFrame(
        {"CV mean": results_mean, "Std": results_std}, index=classifiers
    )
    return models_comparison


def create_confusion_matrix_for_list(
    models_list: List, X_test: np.array, y_test: np.array
) -> plt.figure:
    """Functions that create confusion matrix for machine learning model outcome

    Arg: models_list: list of models.
        X_test: np.array,
        y_test: np.array - numpy arrays with train/test data
    Return: confusion matric - plt.figure
    """
    f, ax = plt.subplots(1, len(models_list), figsize=(24, 4))
    i = 0
    for model in models_list:
        y_pred = cross_val_predict(model, X_test, y_test, cv=6)
        sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax[i], annot=True, fmt="2.0f")
        ax[i].set_title(f"Matrix for \n {model}")
        i += 1
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8
        )


def calculate_predictions(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def create_heatmap(df: pd.DataFrame, size_of_figure: Tuple[int, int]) -> plt.figure:
    """Function creates heatmap of correlation of feature from a given dataframe.

    Arg: df - pd.DataFrame - dataframe with analyzed features
       size_of_figure - Tuple[int] - desired figure size

    Return: plt.figure - heatmap.
    """

    corr_data = df
    corr = corr_data.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(size_of_figure))

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )

    heatmap.set_title(
        f"Correlation heatmap of data attributes",
        fontdict={"fontsize": 16},
        pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")