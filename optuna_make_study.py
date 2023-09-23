import optuna

study = optuna.create_study(storage="sqlite:///example.db", study_name="test2",load_if_exists=True,direction="minimize")
