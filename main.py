# main.py
# Шаг 1: загрузка и подготовка данных

from datasets import load_dataset


def load_recipes(num_samples=10000):
    """
    Загружает датасет рецептов с HuggingFace и подготавливает его для обучения.

    Каждый рецепт оборачивается в специальные токены <|startofrecipe|> и <|endofrecipe|>,
    чтобы модель понимала, где начинается и заканчивается рецепт.

    Args:
        num_samples: количество рецептов для выборки (по умолчанию 50 000)

    Returns:
        Dataset с полем 'text', содержащим отформатированные рецепты
    """
    dataset = load_dataset("corbt/all-recipes", split="train")

    def format_recipe(example):
        text = f"<|startofrecipe|>{example['input']}<|endofrecipe|>"
        return {"text": text}

    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    dataset = dataset.map(format_recipe)

    return dataset


if __name__ == "__main__":
    dataset = load_recipes()
    print(dataset)
    print(dataset[0]["text"][:500])
