# bot.py
# Шаг 4: Gradio-бот

import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def load_model():
    """
    Загрузка дообученной модели и токенизатора из ./recipe_model.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("./recipe_model")
    model = GPT2LMHeadModel.from_pretrained("./recipe_model")
    model.eval()
    return model, tokenizer


def generate_recipe(ingredients):
    """
    Генерация рецепта по введённым ингредиентам.

    Args:
        ingredients: строка с ингредиентами через запятую
    Returns:
        сгенерированный рецепт
    """
    prompt = f"<|startofrecipe|>{ingredients}\n\nIngredients:\n- {ingredients}\n\nDirections:\n-"

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=512,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.encode("<|endofrecipe|>")[0]
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Убираем специальные токены для красивого вывода
    result = result.replace("<|startofrecipe|>", "").replace("<|endofrecipe|>", "").strip()

    return result


# Загрузка модели при старте
model, tokenizer = load_model()

# Создание интерфейса
demo = gr.Interface(
    fn=generate_recipe,
    inputs=gr.Textbox(
        label="Введите ингредиенты",
        placeholder="chicken, garlic, olive oil, salt, pepper",
        lines=3
    ),
    outputs=gr.Textbox(label="Рецепт", lines=15),
    title="Recipe Generator",
    description="Введите ингредиенты через запятую, и модель сгенерирует рецепт."
)

demo.launch()
