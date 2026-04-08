# model_setup.py
# Шаг 2: модель и токенизатор

from transformers import GPT2Tokenizer, GPT2LMHeadModel


def setup_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    special_tokens = {
        "additional_special_tokens": ["<|startofrecipe|>", "<|endofrecipe|>"],
        "pad_token": "<|pad|>"
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = setup_model()

    print(f"Размер словаря: {len(tokenizer)}")
    print(f"Параметров в модели: {model.num_parameters():,}")

    test = "<|startofrecipe|>Add salt<|endofrecipe|>"
    tokens = tokenizer.encode(test)
    print(f"Текст: {test}")
    print(f"Токены: {tokens}")
    print(f"Обратно: {tokenizer.decode(tokens)}")