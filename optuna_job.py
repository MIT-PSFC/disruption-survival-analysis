import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)

    return (x - 2) ** 2


if __name__ == "__main__":

    study = optuna.create_study(
        storage="sqlite:///example.db", 
        study_name="test2",
    load_if_exists=True,direction="minimize")


    study.optimize(objective, n_trials=1000)