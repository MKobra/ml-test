# Подготовка данных для fine-tuning LLaMA-3-8B на русских кулинарных рецептах

## Описание

Скрипт `scripts/01_prepare_data.py` загружает датасет `ZeroAgency/ru-big-russian-dataset`, фильтрует русские кулинарные примеры (topic=cooking, overall_score>=8) и сохраняет в формате JSONL.

## Структура

```
├── data/
│   └── train.jsonl       # 500 примеров instruction + response
├── scripts/
│   └── 01_prepare_data.py
├── requirements.txt
└── README.md
```

## Использование

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py
```
