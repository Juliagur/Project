#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Расширенная система вызова функций с использованием LangChain:
"""

import os

os.environ["HF_HOME"] = r"D:\Юля_уник\5 курс\pr\itogProject\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\Юля_уник\5 курс\pr\itogProject\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from typing import Dict, Any, List
import json
import re
from datetime import datetime

# === 1. Загрузка модели ===
print("🔍 Загружаем модель Phi-3-mini...")
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    use_cache=True
)
model.to("cpu")

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    truncation=True
)

from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=hf_pipe)
print("✅ Модель загружена!\n")


# === 2. MEMORY: Хранилище истории диалога ===
class SimpleMemory:
    def __init__(self):
        self.chat_history: List[str] = []

    def add(self, user: str, ai: str):
        self.chat_history.append(f"Пользователь: {user}")
        self.chat_history.append(f"Агент: {ai}")

    def get_context(self, last_n: int = 4) -> str:
        return "\n".join(self.chat_history[-last_n:]) if self.chat_history else "Нет истории."


memory = SimpleMemory()

# === 3. TOOLS  ===
from langchain_core.tools import tool


@tool
def check_weather(location: str) -> str:
    """Получает текущую погоду в указанном городе."""
    return f"🌤️ В {location} сейчас +22°C, солнечно."


@tool
def book_appointment(date: str, time: str, service: str) -> str:
    """Бронирует запись на услугу."""
    return f"✅ Запись на '{service}' забронирована на {date} в {time}."


@tool
def search_restaurant(cuisine: str, city: str) -> str:
    """Ищет рестораны заданной кухни."""
    return f"🍝 Найдены рестораны {cuisine} кухни в {city}: 'La Bella', 'Taste of Home'."


# TOOL
@tool
def get_current_time(dummy: str = "") -> str:
    """Возвращает текущую дату и время."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    return f"🕒 Текущее время: {now}"


# === 4. RETRIEVAL (база знаний) ===
KNOWLEDGE_BASE = {
    "политика конфиденциальности": "Мы не храним ваши данные. Все запросы обрабатываются локально.",
    "возврат": "Возврат невозможен, так как услуга виртуальная.",
    "поддержка": "Напишите на support@example.com"
}


def retrieve_info(query: str) -> str:
    """Эмуляция Retrieval из базы знаний."""
    query = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query:
            return value
    return None


# === 5. TOOLS + RETRIEVAL ===
tools = [check_weather, book_appointment, search_restaurant, get_current_time]
tool_dict = {tool.name: tool for tool in tools}

# === 6. PROMPT с MEMORY и TOOLS ===
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
Ты — умный ассистент. У тебя есть доступ к следующим функциям:
{tool_descriptions}

История диалога:
{chat_history}

Правила:
1. Если запрос можно выполнить функцией — выведи ТОЛЬКО JSON:
{{"function": "имя", "args": {{"парам": "знач"}}}}
2. Если запрос касается поддержки, возврата, политики — ответь напрямую.
3. Иначе — отвечай самостоятельно.

Пример:
Запрос: Сколько времени?
Ответ: {{"function": "get_current_time", "args": {{}}}}

Запрос: {input}
Ответ:
""".strip()

tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])

# === 7. CHAIN (с наследованием) ===
from langchain_core.chains import Chain


class FunctionCallingChain(Chain):
    """Цепочка с вызовом функций и памятью."""
    llm: Any
    prompt_template: str
    tool_dict: Dict[str, Any]
    memory: Any

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_input = inputs["input"]

        # Извлекаем контекст из памяти
        chat_history = self.memory.get_context()

        # Формируем промпт
        full_prompt = self.prompt_template.format(
            tool_descriptions=tool_descriptions,
            chat_history=chat_history,
            input=user_input
        )

        # Генерация
        raw_output = self.llm.invoke(full_prompt)
        result = self._parse_json(raw_output)

        # Логика вызова
        if result.get("function") == "none":
            response = result.get("response", "Извините, я не понял.")
        else:
            func_name = result.get("function")
            args = result.get("args", {})
            if func_name in self.tool_dict:
                response = self.tool_dict[func_name].func(**args)
            else:
                response = "Неизвестная функция."

        # Проверяем Retrieval
        if "function" not in result or result["function"] == "none":
            retrieved = retrieve_info(user_input)
            if retrieved:
                response = retrieved

        # Сохраняем в память
        self.memory.add(user_input, response)
        return {"output": response}

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group().replace('\n', ' '))
        except Exception:
            pass
        return {"function": "none", "response": "Не удалось понять запрос."}


# Создаём цепочку
chain = FunctionCallingChain(
    llm=llm,
    prompt_template=PROMPT_TEMPLATE,
    tool_dict=tool_dict,
    memory=memory
)


# === 8. ЗАПУСК ===
def run_tests():
    print("🧪 Тесты с Memory и Custom Tool...\n")
    tests = [
        "Сколько времени?",
        "Какая погода в Саратове?",
        "Какая у вас политика конфиденциальности?",
        "Сколько времени?"  # проверка Memory
    ]
    for q in tests:
        result = chain({"input": q})
        print(f"Вопрос: {q}")
        print(f"Ответ: {result['output']}\n")


def main():
    print("🤖 Расширенный агент с Memory, Tools и Chain запущен!")
    print("💬 Введите запрос (или 'выход'):\n")
    while True:
        try:
            user_input = input("Вы: ").strip()
            if not user_input or user_input.lower() in ("выход", "exit"):
                break
            result = chain({"input": user_input})
            print(f"Агент: {result['output']}\n")
        except KeyboardInterrupt:
            print("\n👋 Выход.")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        main()