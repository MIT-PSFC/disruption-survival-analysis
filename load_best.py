import optuna

study = optuna.load_study(
    storage="sqlite:///example.db", 
    study_name="test2")

dict = study.best_params

print(study.best_params)