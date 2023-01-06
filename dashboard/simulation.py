import numpy as np
import pandas as pd
import random

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

n_initial = 1   # Initial number of trained data
n_queries = 50  # Total number of queries to run
n_runs = 5      # Total number of AL runs to perform

def run_simulation(source):

    if source == 'iris':
        data = pd.read_csv("data/iris.csv")
    
    data['class'] = pd.factorize(data['class'])[0] + 1
    X = data.drop('class', axis=1).to_numpy()
    y = data['class'].to_numpy()
    
    total_scores_al = []

    # Runs using AL
    for runs in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

        X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
        X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=uncertainty_sampling,
            X_training=X_initial, y_training=y_initial
        )

        accuracy_scores = [learner.score(X_test, y_test)]

        for i in range(n_queries):
            query_idx, query_inst = learner.query(X_pool)

            y_new = np.array([y_pool[query_idx]], dtype=int)

            learner.teach(query_inst, y_new[0])
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
            accuracy_scores.append(learner.score(X_test, y_test))
        
        total_scores_al.append(accuracy_scores)
        average_scores_al = np.mean(total_scores_al, axis=0)

    total_scores_rand = []

    # Runs using random sampling
    for runs in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

        X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
        X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=uncertainty_sampling,
            X_training=X_initial, y_training=y_initial
        )

        accuracy_scores = [learner.score(X_test, y_test)]

        for i in range(n_queries):
            index_value = random.sample(list(enumerate(X_pool[0])), 1)
            index = index_value[0][0]
            value = index_value[0][1]

            y_new = np.array([y_pool[index]], dtype=int)
            learner.teach([X_pool[index]], y_new)
            X_pool, y_pool = np.delete(X_pool, index, axis=0), np.delete(y_pool, index, axis=0)
            accuracy_scores.append(learner.score(X_test, y_test))
        
        total_scores_rand.append(accuracy_scores)
        average_scores_rand = np.mean(total_scores_rand, axis=0)

    average_scores_al = [[x+1, average_scores_al[x], 'Active Learning'] for x in range(len(average_scores_al))]
    average_scores_rand = [[x+1, average_scores_rand[x], 'Random Learning'] for x in range(len(average_scores_rand))]

    data = average_scores_al + average_scores_rand
    
    output = pd.DataFrame(data, columns=["queries", "accuracy", "type"])

    return output