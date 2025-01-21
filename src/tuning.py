from sklearn.model_selection import GridSearchCV

def tune_random_forest(model, X_train, y_train):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train.values.ravel())

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
