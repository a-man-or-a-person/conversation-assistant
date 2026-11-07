"""Allowlisted Telegram adapter; provider requests are explicitly opt-in."""

import asyncio
import logging
import os
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.filters import Command
from dotenv import load_dotenv

from api import MAX_INPUT_BYTES, ServiceError, suggest_from_image

logger = logging.getLogger("conversation_assistant")


def allowed_user_ids(value: str) -> set[int]:
    try:
        ids = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("ALLOWED_USER_IDS must contain numeric Telegram user IDs.") from exc
    if not ids or any(item <= 0 for item in ids):
        raise ValueError("At least one positive ALLOWED_USER_IDS entry is required.")
    return ids


def authorized(message, users: set[int]) -> bool:
    return (message.chat.type == "private" and message.from_user is not None
            and message.from_user.id in users)


async def safe_send_message(bot: Bot, chat_id: int, text: str) -> bool:
    for offset in range(0, len(text), 2000):
        for attempt in range(3):
            try:
                await bot.send_message(chat_id, text[offset:offset + 2000], parse_mode=None)
                break
            except TelegramRetryAfter as exc:
                if exc.retry_after > 10 or attempt == 2:
                    return False
                await asyncio.sleep(exc.retry_after)
            except TelegramNetworkError:
                if attempt == 2:
                    return False
                await asyncio.sleep(1)
    return True


def build_dispatcher(bot: Bot, users: set[int]) -> Dispatcher:
    dp = Dispatcher()
    lock = asyncio.Lock()

    @dp.message(Command("start"))
    async def start(message):
        if authorized(message, users):
            await safe_send_message(bot, message.chat.id,
                "Отправь изображение переписки, чтобы получить варианты ответа. "
                "Изображение передаётся в Yandex Vision, распознанный текст — в YandexGPT. "
                "Используй только вымышленные переписки или данные, которыми вправе делиться. "
                "Бот не отправляет предложенный ответ собеседнику.")

    @dp.message(F.photo | F.document)
    async def image(message):
        if not authorized(message, users):
            return
        media = message.photo[-1] if message.photo else message.document
        if media.file_size is None or not 0 < media.file_size <= MAX_INPUT_BYTES:
            await safe_send_message(bot, message.chat.id, "Размер изображения — не более 8 МиБ.")
            return
        if lock.locked():
            await safe_send_message(bot, message.chat.id, "Обработка уже идёт. Попробуй чуть позже.")
            return
        async with lock:
            try:
                file = await bot.get_file(media.file_id)
                if file.file_size is None or file.file_size > MAX_INPUT_BYTES:
                    raise ValueError("Image exceeds the size limit.")
                stream = BytesIO()
                await bot.download_file(file.file_path, destination=stream, timeout=30)
                answer = await asyncio.to_thread(suggest_from_image, stream.getvalue())
                await safe_send_message(bot, message.chat.id, answer)
            except (ValueError, ServiceError):
                await safe_send_message(bot, message.chat.id,
                    "Не удалось обработать изображение. Проверь формат и попробуй позже.")
            except Exception:
                logger.warning("Image processing failed.")
                await safe_send_message(bot, message.chat.id, "Сервис временно недоступен.")
    return dp


async def main():
    load_dotenv()
    if os.getenv("LIVE_INTEGRATIONS_ENABLED") != "true":
        raise ValueError("Set LIVE_INTEGRATIONS_ENABLED=true to start live processing.")
    users = allowed_user_ids(os.getenv("ALLOWED_USER_IDS", ""))
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token or not os.getenv("FOLDER_ID") or not os.getenv("YANDEX_AI_API"):
        raise ValueError("Configure the Telegram and Yandex credentials first.")
    logging.basicConfig(level=logging.WARNING)
    # Transport error messages may include request URLs; keep library logging disabled.
    transport_logger = logging.getLogger("aiogram")
    transport_logger.addHandler(logging.NullHandler())
    transport_logger.propagate = False
    session = AiohttpSession(proxy=os.getenv("PROXY_URL") or None)
    async with Bot(token=token, session=session) as bot:
        await build_dispatcher(bot, users).start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
