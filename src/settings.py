logreg_params = dict(multi_class='ovr',
                     class_weight=None,
                     random_state=43,
                     max_iter=300,
                     n_jobs=-1,
                     penalty='l2',
                     C=0.5)

rnn_params = dict(
    # Пути к данным
    df="../coleridgeinitiative-show-us-the-data/train.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="../models",
    # Гиперпараметры архитектуры нейросети
    char_embedding_size=300,
    rnn_hidden_size=64,
    # Гиперпараметры тренировки нейросети
    num_epochs=300,
    learning_rate=1e-2,
    batch_size=32,
    seed=1337,
    early_stopping_criteria=5,
    # Runtime hyper parameter
    cuda=False,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
)