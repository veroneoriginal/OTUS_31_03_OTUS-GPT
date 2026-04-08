# train.py
# Шаг 3: дообучение

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from main import load_recipes
from model_setup import setup_model


def tokenize(example, tokenizer, max_length=256):
    """
    Превращает текст рецепта в числовые токены, понятные модели.

    Длинные рецепты обрезаются до max_length токенов,
    padding не добавляется — это сделает DataCollator при сборке батчей.

    Args:
        example: один пример из датасета с полем 'text'
        tokenizer: токенизатор GPT-2
        max_length: максимальная длина в токенах (по умолчанию 256)

    Returns:
        dict с полями 'input_ids' и 'attention_mask'
    """
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def train():
    """
    Основная функция дообучения GPT-2 на датасете рецептов.

    Процесс:
        1. Загружает 50 000 рецептов и подготовленную модель
        2. Токенизирует все рецепты (текст → числа)
        3. Разбивает данные на train (90%) и validation (10%)
        4. Запускает обучение на 3 эпохи с логированием loss
        5. Сохраняет дообученную модель в ./recipe_model

    Параметры обучения подобраны под GTX 1060 6GB:
        - batch_size=4, fp16=True для экономии видеопамяти
        - max_length=256 токенов на рецепт
        - learning_rate=5e-5 с warmup 500 шагов
    """
    # Загружаем данные и модель
    dataset = load_recipes(num_samples=10000)
    model, tokenizer = setup_model()

    # Токенизируем все рецепты (превращаем текст в числа)
    tokenized = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # Разбиваем на train (90%) /обучающие данные и validation (10%)/проверочные данные
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    # сбор батчей и создание labels для обучения
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Настройки обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        learning_rate=5e-5,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,  # экономим память на GPU
        report_to="none",
    )

    # Создаём тренер и запускаем
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    print("Начинаем обучение...")
    trainer.train()

    # Сохраняем дообученную модель
    model.save_pretrained("./recipe_model")
    tokenizer.save_pretrained("./recipe_model")
    print("Модель сохранена в ./recipe_model")


if __name__ == "__main__":
    train()
