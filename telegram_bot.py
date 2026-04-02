#!/usr/bin/env python3
"""
Telegram bot for the AI Thermostat Agent.
Commands: /status, /history, /export
Natural language messages trigger immediate evaluation.
"""

import asyncio
import logging
from telegram import Update
from telegram.error import TimedOut, NetworkError
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

import agent
import database

logger = logging.getLogger(__name__)


async def _safe_reply(message, text: str, retries: int = 1):
    """Send a reply with retry on timeout. Splits long messages at 4096 chars."""
    MAX_LEN = 4096
    if len(text) <= MAX_LEN:
        chunks = [text]
    else:
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > MAX_LEN:
                if current:
                    chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)

    for chunk in chunks:
        for attempt in range(retries + 1):
            try:
                await message.reply_text(chunk)
                break
            except (TimedOut, NetworkError) as e:
                if attempt < retries:
                    logger.warning("Telegram reply timed out, retrying...")
                    await asyncio.sleep(2)
                else:
                    logger.error("Telegram reply failed after %d attempts: %s", retries + 1, e)

_app: Application = None
_config: dict = {}
_whitelisted_ids: set = set()


def init_bot(config: dict):
    """Initialize the Telegram bot (called by agent.main)."""
    global _config, _whitelisted_ids
    _config = config
    _whitelisted_ids = set(config["telegram"].get("whitelisted_chat_ids", []))
    logger.info("Telegram bot initialized (whitelisted IDs: %s)", _whitelisted_ids)


def _is_authorized(chat_id: int) -> bool:
    """Check if a chat ID is whitelisted."""
    if not _whitelisted_ids or 0 in _whitelisted_ids:
        return True  # No whitelist configured or placeholder value
    return chat_id in _whitelisted_ids


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command — show current thermostat state and last decision."""
    if not _is_authorized(update.effective_chat.id):
        return

    try:
        import nest_api
        import weather as weather_mod

        states = nest_api.get_all_thermostat_states()
        w = weather_mod.get_weather()
        heartbeat = database.get_heartbeat()
        last_result = agent.get_last_evaluation_result()

        lines = ["Thermostat Status", "=" * 25]

        for state in states:
            lines.append(f"\n{state.name}:")
            lines.append(f"  Indoor: {state.indoor_temp}F, {state.humidity}% humidity")
            lines.append(f"  Mode: {state.mode}, Target: {state.target_temp}F")
            lines.append(f"  HVAC: {'running' if state.hvac_running else 'idle'}")

        lines.append(f"\nOutdoor: {w.current_temp}F, {w.humidity}% humidity")
        lines.append(f"Forecast: {w.forecast_summary}")

        if w.is_stale:
            lines.append("(Weather data is stale)")

        if last_result:
            lines.append("")
            zone = last_result.get("zone", "")
            lines.append(f"Last decision{f' ({zone})' if zone else ''}:")
            lines.append(f"  {last_result.get('action', 'unknown')}")
            if last_result.get("temperature"):
                lines.append(f"  Set to: {last_result['temperature']}F")
            lines.append(f"  {last_result.get('reasoning', 'N/A')}")

        if heartbeat:
            lines.append(f"\nLast cycle: {heartbeat}")

        await _safe_reply(update.message,"\n".join(lines))

    except Exception as e:
        logger.error("Status command failed: %s", e)
        await _safe_reply(update.message,f"Error getting status: {e}")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /history command — show recent decisions."""
    if not _is_authorized(update.effective_chat.id):
        return

    try:
        decisions = database.get_recent_decisions(2)

        if not decisions:
            await _safe_reply(update.message,"No decisions recorded yet.")
            return

        lines = []
        for d in decisions:
            ts = d["timestamp"][:16]
            zone = d.get("zone", "")
            if d["action"] == "set_temperature":
                lines.append(f"{ts} [{zone}] set to {d['temperature']:.0f}F")
            else:
                lines.append(f"{ts} [{zone}] unchanged")
            lines.append(f"  {(d.get('reasoning') or '')[:200]}")

        await _safe_reply(update.message,"\n".join(lines))

    except Exception as e:
        logger.error("History command failed: %s", e)
        await _safe_reply(update.message,f"Error getting history: {e}")


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /export command — export climate log as CSV."""
    if not _is_authorized(update.effective_chat.id):
        return

    try:
        csv_data = database.export_climate_csv()

        if csv_data == "No data":
            await _safe_reply(update.message,"No climate data to export yet.")
            return

        # Send as a file
        import io
        bio = io.BytesIO(csv_data.encode("utf-8"))
        bio.name = "climate_log.csv"
        await update.message.reply_document(document=bio,
                                             caption="Climate log export")

    except Exception as e:
        logger.error("Export command failed: %s", e)
        await _safe_reply(update.message,f"Error exporting data: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle natural language messages.
    Log the message and trigger an immediate evaluation cycle.
    """
    if not _is_authorized(update.effective_chat.id):
        return

    text = update.message.text
    chat_id = update.effective_chat.id

    logger.info("Message from %s: %s", chat_id, text)

    # Log the message
    database.log_message(chat_id, text)

    # Acknowledge receipt
    await _safe_reply(update.message,"Got it. Running evaluation...")

    # Record current counter, then trigger
    old_counter = agent.get_evaluation_counter()
    agent.trigger_evaluation()

    # Wait for a NEW cycle to complete (up to 3 minutes)
    result = None
    for _ in range(36):  # 36 x 5s = 180s max wait
        await asyncio.sleep(5)
        if agent.get_evaluation_counter() > old_counter:
            result = agent.get_last_evaluation_result()
            break

    if result:
        # Prefer message_to_user if the LLM wrote one (e.g. answering a question)
        llm_message = result.get("message_to_user")
        if llm_message:
            await _safe_reply(update.message,llm_message)
            database.log_message(chat_id, text, llm_message)
        else:
            response_parts = []
            zone = result.get("zone", "")
            if zone:
                response_parts.append(f"[{zone}]")
            if result["action"] == "set_temperature":
                response_parts.append(f"Set to {result['temperature']}F")
            else:
                response_parts.append("No change needed")
            response_parts.append(result.get("reasoning", ""))

            response_text = " — ".join(response_parts)
            await _safe_reply(update.message,response_text)
            database.log_message(chat_id, text, response_text)
    else:
        # Fallback: show current status even if evaluation failed
        try:
            import nest_api as _nest
            import weather as _weather
            states = _nest.get_all_thermostat_states()
            w = _weather.get_weather()
            lines = []
            for s in states:
                lines.append(f"{s.name}: {s.indoor_temp}F, {s.humidity}% humidity, "
                             f"mode={s.mode}, target={s.target_temp}F")
            lines.append(f"Outdoor: {w.current_temp}F")
            lines.append("(Evaluation didn't produce a decision — showing current status)")
            await _safe_reply(update.message,"\n".join(lines))
        except Exception:
            await _safe_reply(update.message,
                "Evaluation didn't complete. Try /status for current readings."
            )


async def send_agent_message(message: str):
    """Send an outbound message to all whitelisted chats (alerts, reports).
    Splits messages longer than 4096 chars (Telegram API limit).
    """
    if not _app or not _app.bot:
        logger.warning("Cannot send message — bot not initialized (app=%s)", _app)
        return

    # Split into chunks at newlines, respecting Telegram's 4096-char limit
    MAX_LEN = 4096
    chunks = []
    if len(message) <= MAX_LEN:
        chunks = [message]
    else:
        current = ""
        for line in message.split("\n"):
            if len(current) + len(line) + 1 > MAX_LEN:
                if current:
                    chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)

    logger.info("Sending to %d chat(s) (%d chunk(s)): %s",
                len(_whitelisted_ids), len(chunks), message[:60])
    for chat_id in _whitelisted_ids:
        if chat_id == 0:
            continue  # Skip placeholder
        for chunk in chunks:
            try:
                await _app.bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as e:
                logger.error("Failed to send to %s: %s", chat_id, e)


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors from the Telegram bot without spamming the console."""
    logger.error("Telegram bot error: %s", context.error)


async def start_bot():
    """Start the Telegram bot polling (called by agent.main via asyncio.gather)."""
    global _app

    token = _config["telegram"]["bot_token"]
    if token == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.warning("Telegram bot token not configured — bot disabled")
        # Keep running so asyncio.gather doesn't exit
        while True:
            await asyncio.sleep(3600)

    _app = Application.builder().token(token).build()

    # Register handlers
    _app.add_handler(CommandHandler("status", cmd_status))
    _app.add_handler(CommandHandler("history", cmd_history))
    _app.add_handler(CommandHandler("export", cmd_export))
    _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    _app.add_error_handler(_error_handler)

    logger.info("Telegram bot starting polling...")

    # Initialize and start
    await _app.initialize()
    await _app.start()

    # Register the send function NOW — bot is ready to send messages
    # (start_polling is only needed for RECEIVING, not sending)
    agent.set_telegram_send_fn(send_agent_message)
    logger.info("Telegram send function registered with agent")

    await _app.updater.start_polling(
        drop_pending_updates=True
    )

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await _app.updater.stop()
        await _app.stop()
        await _app.shutdown()
