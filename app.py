import random
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import speech_recognition as sr
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from config import BOT_CONFIG
from database import get_random_product, get_product_by_category
from io import BytesIO


X_text = []  
y = []  

print('читаю bot_config')
for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)

print('Использую машинное обучение')
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)
train_accuracy = clf.score(X, y)
print(f"Точность (метод score): {train_accuracy:.2f}")

def clear_phrase(phrase):
    phrase = phrase.lower()

    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    result = ''.join(symbol for symbol in phrase if symbol in alphabet)
    
    return result.strip()

def classify_intent(replica):
    replica = clear_phrase(replica)

    intent = clf.predict(vectorizer.transform([replica]))[0]
   
    for example in BOT_CONFIG['intents'][intent]['examples']:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent

def get_answer_by_intent(intent, is_audio = False):
    if intent in BOT_CONFIG['intents']:
        if intent == 'tables' and not is_audio:
            product = get_random_product()
            if product:
                return {
                    'type': 'product',
                    'data': product
                }
        elif intent in ['thin', 'corner'] and not is_audio:  
            product = get_product_by_category(intent)
            if product:
                return {
                    'type': 'product',
                    'data': product  
                }
        responses = BOT_CONFIG['intents'][intent]['responses']
        if responses:
            return random.choice(responses)
print('Открываю диалоги')
with open('dialogs.txt',encoding='utf-8') as f:
    content = f.read()

dialogues_str = content.split('\n\n')
dialogues = [dialogue_str.split('\n')[:2] for dialogue_str in dialogues_str]

dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue
   
    question, answer = dialogue
    question = clear_phrase(question[2:])
    answer = answer[2:]
   
    if question != '' and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])

dialogues_structured = {}  #  {'word': [['...word...', 'answer'], ...], ...}

for question, answer in dialogues_filtered:
    words = set(question.split(' '))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])

dialogues_structured_cut = {}
for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]

def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(' '))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]
    mini_dataset = list(set(mini_dataset))

    answers = [] 

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])
   
    if answers:
        return min(answers, key=lambda three: three[0])[2]

def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)

stats = {'intent': 0, 'generate': 0, 'failure': 0}

def recognise_audio(file_path):
    r = sr.Recognizer()
    audio_file = file_path
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source) # Запись аудио из файла
        # Распознавание речи
    try:
        text = r.recognize_google(audio, language="ru-RU")
        print("Распознанный текст:", text)
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print("Ошибка сервиса распознавания речи; {0}".format(e))

def generate_audio(message):
    tts = gTTS(text=message, lang='ru')
    tts.save("output_temp.mp3")

def generate_audio_response(message):
    intent = classify_intent(message)

    # Answer generation
   
    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent,1)
        if answer:
            stats['intent'] += 1
            print("audio anser intent")
            generate_audio(answer)
            return

    # вызов генеративной модели
    answer = generate_answer(message)
    if answer:
        stats['generate'] += 1
        print("audio answer generate")
        generate_audio(answer)
        return
   
    # берем заглушку
    stats['failure'] += 1
    print("audio answer fail")
    generate_audio(get_failure_phrase())
    print(answer)
    return

def bot(replica):
    # NLU
    intent = classify_intent(replica)

    # Answer generation
   
    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1
            return answer

    # вызов генеративной модели
    answer = generate_answer(replica)
    if answer:
        stats['generate'] += 1
        return answer
   
    # берем заглушку
    stats['failure'] += 1
    return get_failure_phrase()


    

from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext,  ApplicationBuilder
import string
import requests
import os
import base64
from dotenv import load_dotenv
API_URL = "http://localhost:7860/sdapi/v1/txt2img"

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
api_key = os.getenv('API_KEY')  

async def generate_image(update: Update, context: CallbackContext):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Пожалуйста, укажите промпт для генерации изображения.")
        return
    
    payload = {
        "prompt": prompt,
        "steps": 30,
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json() 
        image_data = base64.b64decode(data["images"][0])  
        image = BytesIO(image_data)
        image.seek(0)
        await update.message.reply_photo(photo=image)
        print(f'Изображение сгенерированно prompt:{prompt}')
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка при генерации изображения: {str(e)}")
        print(f'Ошибка при генерации изображения: {str(e)}')

async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await  update.message.reply_text('Привет! Я могу разговаривать текстом и голосовыми!\nТакже могу выполнять команды\n/choose - выбрать случайный вариант\n/password - сгенерировать пароль\n/generate - сгенерировать изображение')

async def choose(update: Update, context: CallbackContext):
    options = context.args  # Список переданных аргументов
    if not options:
        await update.message.reply_text("Пример: /choose пицца суши паста")
        return
    choice = random.choice(options)
    await update.message.reply_text(f"🎲 Мой выбор: {choice}!")

async def password(update: Update, context: CallbackContext):
    try:
        length = int(context.args[0]) if context.args else 12  # По умолчанию 12 символов
        if length < 4:
            await update.message.reply_text("Длина пароля должна быть ≥ 4")
            return
        
        # Создаём пароль: буквы + цифры + спецсимволы
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        generated_password = ''.join(random.choice(chars) for _ in range(length))
        await update.message.reply_text(f"🔑 Ваш пароль: `{generated_password}`", parse_mode="Markdown")
    
    except (IndexError, ValueError):
        await update.message.reply_text("Пример: /password 10")

async def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text('Для общения просто напишите сообщение, или отправьте голосовое!\nкоманда /choose имеет следующий синтаксис:\n/choose арг1 арг2 арг3\nкоманда /password имеет следующий синтаксис (цифра - длина пароля)\n/password 12\nкоманда /generate имеет синтаксис:\n/generate ПримерПромпта')

async def text_handler(update: Update, context: CallbackContext) -> None:
    replica = update.message.text
    answer = bot(replica)
    if isinstance(answer, dict) and answer.get('type') == 'product':
        product = answer['data']
        # Загружаем изображение по URL
        response = requests.get(product['image_url'])
        photo = BytesIO(response.content)
        photo.name = 'product.jpg'
        
        caption = f"Могу порекомендовать: {product['name']}\n\n{product['description']}\n\nЦена: {product['price']} руб."
        await update.message.reply_photo(photo=photo, caption=caption)
    else:
        await update.message.reply_text(answer)
   
    print(stats); print(replica); print(answer); print()

async def get_voice(update: Update, context: CallbackContext) -> None:
    """Обработка аудио сообщений"""
    new_file = await context.bot.get_file(update.message.voice.file_id)
    await new_file.download_to_drive("audio_temp.ogg")
    data, samplerate = sf.read('audio_temp.ogg')
    sf.write('audio_temp.wav', data, samplerate)
    message = recognise_audio('audio_temp.wav')
    if message:
        generate_audio_response(message)
        my_file = os.path.abspath("output_temp.mp3")
        sound = AudioSegment.from_file(my_file)
        sound.export("output_temp.ogg", format="ogg")
        await update.message.reply_audio("output_temp.ogg")
        print(stats); print(message); print("audio response"); print()
    os.remove("output_temp.ogg")
    os.remove('output_temp.mp3')  
    os.remove("audio_temp.ogg")
    os.remove('audio_temp.wav')

    


def main():
    """Start the bot."""
    application = ApplicationBuilder().token(api_key).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT  & ~ filters.COMMAND, text_handler))
    application.add_handler(MessageHandler(filters.VOICE, get_voice))
    application.add_handler(CommandHandler("password", password))
    application.add_handler(CommandHandler("choose", choose))
    application.add_handler(CommandHandler("generate", generate_image))
    
    # Start the Bot
    print("Запуск приложения")
    application.run_polling()

main()