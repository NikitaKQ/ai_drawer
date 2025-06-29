import asyncio
import logging
import os
import signal
import sys
from io import BytesIO

import requests
from openai import OpenAI
from PIL import Image
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def _download_file(file_url: str) -> BytesIO:
    """Download a file and return bytes buffer."""
    resp = requests.get(file_url)
    resp.raise_for_status()
    return BytesIO(resp.content)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Хай! Кинь текст, картинку или combo 'картинка+подпись'. Я вернусь с новой пикчей от DALL·E‑3 ✨"
    )


async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    prompt = message.caption or message.text or ""

    await message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)

    try:
        if message.photo:
            logger.info("Photo received. Caption: %s", prompt)
            file_obj = await message.photo[-1].get_file()
            photo_bytes = await _download_file(file_obj.file_path)
            
            photo_bytes.seek(0)
            img = Image.open(photo_bytes).convert("RGBA")
            png_bytes = BytesIO()
            img.save(png_bytes, format="PNG")
            png_bytes.seek(0)
            
            image_file = ("image.png", png_bytes, "image/png")

            if prompt:
                response = openai_client.images.edit(
                    model="gpt-image-1",
                    prompt=prompt,
                    image=image_file,
                    n=1,
                    size="1024x1024",
                )
            else:
                response = openai_client.images.create_variation(
                    model="gpt-image-1",
                    image=image_file,
                    n=1,
                    size="1024x1024",
                )
        else:
            prompt = prompt.strip()
            if not prompt:
                await message.reply_text("Эээ… Мне нужен хотя бы текст или картинка!")
                return
            logger.info("Prompt: %s", prompt)
            response = openai_client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=1,
                size="1024x1024",
                style="vivid",
            )

        image_url = response.data[0].url
        logger.info("Generated image URL: %s", image_url)

        image_bytes = await _download_file(image_url)
        image_bytes.seek(0)
        await message.reply_photo(photo=image_bytes)

    except Exception as e:
        logger.exception("Generation failed: %s", e)
        await message.reply_text("🚨 Упс, что‑то пошло не так. Попробуй ещё раз позже!")


async def main():
    """Main function that properly initializes and runs the bot."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token or not os.getenv("OPENAI_API_KEY"):
        logger.error("▶️ Перед стартом задай переменные окружения: OPENAI_API_KEY и TELEGRAM_BOT_TOKEN")
        return

    application = ApplicationBuilder().token(bot_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO | filters.TEXT, process_media))

    logger.info("Bot up and running… Press Ctrl‑C to stop.")
    
    await application.initialize()
    await application.start()
    
    await application.updater.start_polling()
    
    stop_signals = (signal.SIGTERM, signal.SIGINT)
    for sig in stop_signals:
        signal.signal(sig, lambda s, f: asyncio.create_task(shutdown(application)))
    
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await shutdown(application)


async def shutdown(application):
    """Gracefully shutdown the bot."""
    logger.info("Shutting down bot...")
    await application.updater.stop()
    await application.stop()
    await application.shutdown()


def run_bot():
    """Entry point that handles the event loop properly."""
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(main())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()
                
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise


if __name__ == "__main__":
    run_bot()
