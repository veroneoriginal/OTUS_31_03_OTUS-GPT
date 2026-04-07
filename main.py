from datasets import load_dataset

# Скачиваю датасет
dataset = load_dataset("corbt/all-recipes", split="train")

# Оборачиваю каждый рецепт в токены
def format_recipe(example):
    text = f"<|startofrecipe|>{example['input']}<|endofrecipe|>"
    return {"text": text}

# Беру 50 000 рецептов
dataset = dataset.shuffle(seed=42).select(range(50000))
dataset = dataset.map(format_recipe)

# Проверяю результат
print(dataset)
print(dataset[0]["text"][:500])
