import os
import asyncio
import logging
import numpy as np
import torch
import librosa
import joblib 
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from transformers import AutoFeatureExtractor, AutoModel

API_TOKEN = '8438809962:AAHBHiaoCB_WiXDqDhQPDX9XV6dpuogdH-8' 
MODEL_NAME = "ntu-spml/distilhubert"
CLASSIFIER_PATH = "model/hubert_emb_clf.joblib" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS_MAP = {
    0: "🇷🇺 Русский язык",
    1: "🇬🇧 Английский язык",
    2: "🇪🇸 Испанский язык",
    3: "🇫🇷 Французский язык",
    4: "🇩🇪 Немецкий язык"
}

logging.basicConfig(level=logging.INFO)

print(f"Загрузка Hubert-модели на {DEVICE}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(DEVICE)

if os.path.exists(CLASSIFIER_PATH):
    print(f"Загрузка классификатора из {CLASSIFIER_PATH}...")
    classifier = joblib.load(CLASSIFIER_PATH)
else:
    print(f"Файл {CLASSIFIER_PATH} не найден!")
    classifier = None

def load_audio(path):
    y, _ = librosa.load(path, sr=16000)
    return y

@torch.no_grad()
def extract_features(path):
    y = load_audio(path)

    inputs = feature_extractor(
        y,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    
    layer = hidden_states[-1] 
    
    feats = layer.squeeze(0).cpu().numpy()  
    
    mean = feats.mean(axis=0)
    return mean

def get_prediction(path):
    features = extract_features(path)
    
    if classifier is None:
        raise ValueError("Классификатор не загружен")

    features_reshaped = features.reshape(1, -1)
    
    prediction = classifier.predict(features_reshaped)[0]
    
    return prediction

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот-классификатор аудио. 🤖\n\n"
        "Пришли мне голосовое сообщение, я пропущу его через предобученную модель классификации"
        f"и скажу, на каком языке это сообщение!"
    )

@dp.message(F.voice)
async def handle_voice(message: types.Message):
    status_msg = await message.reply("🎧 Слушаю и анализирую...")
    
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    temp_filename = f"voice_{file_id}.ogg"

    try:
        await bot.download_file(file.file_path, temp_filename)

        result_class = await asyncio.to_thread(get_prediction, temp_filename)

        label_text = LABELS_MAP.get(result_class, str(result_class))

        await status_msg.edit_text(
            f"*Анализ завершен!*\n\n"
            f"**Результат:** `{label_text}`",
            parse_mode="Markdown"
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        await status_msg.edit_text("Произошла ошибка при классификации.")
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")
