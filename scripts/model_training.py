import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.catboost
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import git

# ==============================================================================
# 1. НАСТРОЙКА MLFLOW И ПАРАМЕТРОВ ЭКСПЕРИМЕНТА
# ==============================================================================

# Указываем MLflow, куда отправлять данные
mlflow.set_tracking_uri("http://213.21.252.250:5000")

# Задаем имя эксперимента
mlflow.set_experiment("LSTM (test)")

# --- Получаем хеш коммита Git ---
try:
    repo = git.Repo(search_parent_directories=True)
    git_commit_hash = repo.head.object.hexsha
except Exception as e:
    git_commit_hash = "N/A" # На случай, если скрипт запущен не из Git-репозитория
    print(f"Warning: Could not get git commit hash. {e}")

print(f"Current Git Commit Hash: {git_commit_hash}")

# --- Параметры, которые нужно логировать ---
# Параметры из скрипта нарезки данных (sample_creator)
data_params = {
    "window_size": 50,
    "step": 1,
    "sampling_rate": 10
}

# Гиперпараметры модели
model_params = {
    "epochs": 1,
    "batch_size": 512,
    "validation_split": 0.2,
    "optimizer": "adam",
    "loss": "mean_squared_error"
}


# ==============================================================================
# 2. ОСНОВНОЙ КОД ОБУЧЕНИЯ (обернут в MLflow)
# ==============================================================================

# Запускаем новый "прогон" эксперимента в MLflow
with mlflow.start_run():
    print("Starting MLflow run...")

    # --- Логируем параметры ---
    mlflow.log_params(data_params)
    mlflow.log_params(model_params)
    mlflow.set_tag("git_commit", git_commit_hash)
    print("Parameters logged.")

    # --- Загрузка и подготовка данных (ваш код) ---
    def load_and_merge_data(npz_units):
        sample_array_lst = []
        label_array_lst = []
        for npz_unit in npz_units:
          loaded = np.load(npz_unit)
          sample_array_lst.append(loaded['sample'])
          label_array_lst.append(loaded['label'])
        sample_array = np.dstack(sample_array_lst)
        label_array = np.concatenate(label_array_lst)
        sample_array = sample_array.transpose(2, 0, 1)
        return sample_array, label_array

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(project_dir, 'data', 'processed')

    # Собираем пути к файлам для train и test
    train_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.startswith(('Unit2_', 'Unit5_', 'Unit10_', 'Unit16_', 'Unit18_', 'Unit20_'))]
    test_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.startswith(('Unit11_', 'Unit14_', 'Unit15_'))]
    print(train_files)

    # Загружаем данные
    X_train, y_train = load_and_merge_data(train_files)
    X_test, y_test = load_and_merge_data(test_files)

    print('Размер обучающей выборки (X):', X_train.shape)
    print('Размер обучающей выборки (y):', y_train.shape)
    print('Размер тестовой выборки (X):', X_test.shape)
    print('Размер тестовой выборки (y):', y_test.shape)

    # --- Обучение модели (ваш код) ---
    # Определяем форму входных данных из X_train
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

    # Создаем простую LSTM модель
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True)) # return_sequences=True, если следующий слой тоже LSTM
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1)) # Один выход, так как предсказываем одно число - RUL

    # Компилируем модель
    model.compile(optimizer=model_params['optimizer'], loss=model_params['loss'], metrics=['mae'])

    # "Перехватываем" вывод model.summary() в строку
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    model_summary_string = "\n".join(summary_list)

    print(model.summary())

    # Обучаем модель
    history = model.fit(X_train, y_train, 
                        epochs=model_params['epochs'], 
                        batch_size=model_params['batch_size'], 
                        validation_split=model_params['validation_split'], # Используем часть данных для валидации на лету
                        callbacks=[mlflow.keras.MLflowCallback()], # Автоматическое логирование
                        verbose=1)

    # --- Оценка модели и логирование метрик ---
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest MAE: {mae:.2f}')

    metrics = {
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    print(f"Metrics logged: {metrics}")

    # --- Логирование самой модели ---
    mlflow.keras.log_model(
        model,
        artifact_path="lstm-model", # Название папки с моделью в MLflow
    )
    print("Model logged as an artifact.")

    print("MLflow run finished successfully!")
