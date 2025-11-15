# turbo-predict

Проект по прогнозированию работы газотурбинных установок, выполненный в рамках 5-го семестра.

***

## Описание проекта

Этот репозиторий содержит код и материалы, связанные с задачами анализа и прогнозирования работы газотурбинных установок. В проекте реализованы методы нарезки данных, обучения моделей и оценки их качества.

***

## Установка виртуального окружения

Для установки необходимых зависимостей используйте команду:

```bash
poetry install --no-root
```

Это создаст необходимое окружение с библиотеками, указанными в файле `pyproject.toml`.

## Работа с хранилищем s3

1. Установите AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#getting-started-install-instructions)

2. Откройте терминал и выполните команду aws configure:

```bash
aws configure
```

3. Последовательно введите данные:

- AWS Access Key ID: Вставьте Идентификатор ключа.
- AWS Secret Access Key: Вставьте Секретный ключ.
- Default region name: Введите ru-central1.
- Default output format: Просто нажмите Enter (оставьте пустым).

4. Скопируйте следующий текст в файл `~/.aws/config`:

```
[default]
region = ru-central1

[plugins]
endpoint = awscli_plugin_endpoint

[profile default]
s3 =
  endpoint_url = https://storage.yandexcloud.net
  signature_version = s3v4
s3api =
  endpoint_url = https://storage.yandexcloud.net
```

***

## Работа с данными

### Нарезка данных

Процесс подготовки данных включает создание обучающих и тестовых выборок с помощью следующей команды:

1. Создание обучающих выборок
```bash
python -m scripts.sample_creator_unit_auto -w 50 -s 1 --test 0 --sampling 10
```

2. Создание тестовых выборок
```bash
python -m scripts.sample_creator_unit_auto -w 50 -s 1 --test 1 --sampling 10
```

Объяснение параметров:

- `-w 50` — размер окна для нарезки данных.
- `-s 1` — шаг сдвига окна.
- `--test 0` или `1` — режим работы (обучение или тестирование).
- `--sampling 10` — частота выборки.

***

## Обучение и тестирование моделей

### Объекты для обучения и тестирования

- **Обучающие единицы (N=6):** u = 2, 5, 10, 16, 18, 20
- **Тестовые единицы (M=3):** u = 11, 14, 15