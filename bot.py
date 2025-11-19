from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CallbackQueryHandler, filters

import whisper

from schema import ExtractionResult
import llm
from utils import audio
import todoist_client
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Dict, List, Optional

load_dotenv()

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bot")
SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "180"))

_whisper_model = None


def _ensure_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    model_size = os.getenv("WHISPER_MODEL", "medium").strip() or "medium"
    try:
        _whisper_model = whisper.load_model(model_size)
        logger.info("OpenAI Whisper model loaded: %s", model_size)
    except Exception as e:
        logger.exception("Failed to load Whisper model: %s", e)
        _whisper_model = None
    return _whisper_model

def _to_utc_z(dt_str: str | None, *, force_local: bool = False) -> str | None:
    """Interpret dt_str as local time in USER_TIMEZONE and convert to UTC Z.
    Accepts ISO strings with or without timezone; if offset present, use it; otherwise assume USER_TIMEZONE.
    Returns RFC3339 with trailing 'Z'."""
    if not dt_str:
        return None
    s = dt_str.strip()
    if not s:
        return None
    try:
        # Normalize 'Z' to '+00:00' for parsing
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        # If tzinfo absent OR we should force interpret as local (based on original input)
        if dt.tzinfo is None or force_local:
            # Assume user's timezone
            try:
                from zoneinfo import ZoneInfo
                tzname = os.getenv("USER_TIMEZONE", "UTC") or "UTC"
                tz = ZoneInfo(tzname)
            except Exception:
                tz = timezone.utc
            # If datetime already had tz but we force local, drop it before assigning
            dt = dt.replace(tzinfo=None).replace(tzinfo=tz)
        # Convert to UTC
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return s  # fallback: return as-is

def _to_local_naive(dt_str: str | None) -> str | None:
    """Return a naive ISO string preserving the wall-clock time the model produced.
    Rules:
    - If input has timezone (Z or +HH:MM), DO NOT convert — just drop tz and keep HH:MM:SS.
    - If input has no timezone, return as-is (normalized, no microseconds).
    This makes the stored time in Todoist match the preview time exactly.
    """
    if not dt_str:
        return None
    s = dt_str.strip()
    if not s:
        return None
    try:
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        # If has tzinfo, drop it without converting to preserve wall-clock time
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        # If no tzinfo, just normalize by removing microseconds
        return dt.replace(microsecond=0).isoformat()
    except Exception:
        return s

def _format_local_wall(dt_str: str | None, *, original_input: str = "") -> str:
    """Format datetime string as local wall time without tz (for preview).
    If parsing fails, return original string.
    """
    if not dt_str:
        return "—"
    s = dt_str.strip()
    if not s:
        return "—"
    try:
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        force_local = _should_force_local_from_input(original_input)
        try:
            from zoneinfo import ZoneInfo
            tzname = os.getenv("USER_TIMEZONE", "UTC") or "UTC"
            local_tz = ZoneInfo(tzname)
        except Exception:
            local_tz = timezone.utc
        if dt.tzinfo is not None:
            # Если во входе есть таймзона (например, Z/UTC), всегда конвертируем в локальную,
            # чтобы предпросмотр отражал локальное время. Не "сохраняем настенные часы".
            dt = dt.astimezone(local_tz)
        # Drop tz for preview; keep wall-clock time
        return dt.replace(tzinfo=None, microsecond=0).isoformat()
    except Exception:
        return s

def _to_local_with_offset(dt_str: str | None, *, force_local: bool = False) -> str | None:
    """Return ISO string with explicit local timezone offset from USER_TIMEZONE.
    Ensures wall-clock time aligns with preview in Todoist UI.
    Examples (Europe/Moscow, +03:00):
      2025-11-15T14:00:00 -> 2025-11-15T14:00:00+03:00
      2025-11-15T11:00:00Z -> 2025-11-15T14:00:00+03:00
    """
    if not dt_str:
        return None
    s = dt_str.strip()
    if not s:
        return None
    try:
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        # If tzinfo missing, assume local tz; otherwise convert to local tz
        try:
            from zoneinfo import ZoneInfo
            tzname = os.getenv("USER_TIMEZONE", "UTC") or "UTC"
            local_tz = ZoneInfo(tzname)
        except Exception:
            local_tz = timezone.utc
        if dt.tzinfo is None:
            # Наивное время трактуем как локальное (намерение пользователя) — проставляем локальную TZ
            dt = dt.replace(tzinfo=local_tz)
        else:
            # Если во входе есть TZ (например, UTC), всегда переводим момент времени в локальную TZ
            dt = dt.astimezone(local_tz)
        return dt.replace(microsecond=0).isoformat()
    except Exception:
        return s

def _should_force_local_from_input(text: str) -> bool:
    t = (text or "").lower()
    # If user mentioned explicit tz markers, don't force
    if any(x in t for x in ["utc", "gmt", "+0", "+1", "+2", "+3", "-0", "-1", "-2", "-3", "z"]):
        return False
    # Heuristics: relative words or explicit time without tz usually mean local intent
    rel_words = ["сегодня", "завтра", "послезавтра", "today", "tomorrow"]
    if any(w in t for w in rel_words):
        return True
    # Simple HH:MM pattern presence
    import re
    if re.search(r"\b\d{1,2}:\d{2}\b", t):
        return True
    return False


async def _check_user_access(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is authorized. If not, send spooky refusal and return False."""
    allowed_id = os.getenv("ALLOWED_USER_ID", "").strip()
    if not allowed_id:
        # If not configured, allow everyone (backward compatibility)
        return True
    user = update.effective_user
    if user and str(user.id) == allowed_id:
        return True
    # Unauthorized user - generate and send refusal message
    user_name = user.first_name if user else "незнакомец"
    try:
        refusal_msg = llm.generate_refusal(user_name)
    except Exception:
        refusal_msg = "Уходи... Тебе здесь не рады... 👻"
    await update.effective_message.reply_text(refusal_msg)
    logger.warning(f"Unauthorized access attempt from user_id={user.id} ({user_name})")
    return False


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Check access first
    if not await _check_user_access(update, context):
        return
    
    message = update.effective_message
    text = message.text or ""
    # Любая активность пользователя — обновляем таймер сессии
    _touch_activity(context, update.effective_chat.id, update.effective_user.id)

    # Если предыдущая сессия истекла и пришёл НОВЫЙ свободный текст — начинаем заново
    if context.user_data.get("session_expired"):
        # Сбрасываем старое состояние, если это не кнопка/не ответы к старой сессии
        context.user_data.clear()
    if not text.strip():
        await message.reply_text("Пустое сообщение.")
        return
    # Если ожидаем ввод для редактирования предпросмотра
    if context.user_data.get("awaiting_refine_input"):
        pending = context.user_data.get("pending_result")
        original_input = context.user_data.get("original_input", "")
        if not pending:
            context.user_data["awaiting_refine_input"] = False
            await message.reply_text("Нет данных для редактирования. Отправьте новое сообщение.")
            return
        updated = llm.refine_tasks(original_input, pending, text)
        try:
            updated = llm.validate_extraction(original_input, updated)
        except Exception:
            pass
        context.user_data["pending_result"] = updated
        context.user_data["awaiting_refine_input"] = False
        await _show_preview(message, context, updated, original_input)
        return
    # Если у нас ожидаются ответы на уточняющие вопросы
    user_state = context.user_data.get("awaiting_clarifications")
    if user_state:
        original_input: str = context.user_data.get("original_input", "")
        combined = original_input + "\n\nОтветы пользователя на уточняющие вопросы:\n" + text
        result: ExtractionResult = llm.extract_tasks(combined)
        # Сброс состояния
        context.user_data["awaiting_clarifications"] = False
        context.user_data["original_input"] = ""
        if result.clarifying_questions:
            # В случае если всё ещё не хватает данных — задаём ещё раз
            qs = _format_questions(result.clarifying_questions)
            context.user_data["awaiting_clarifications"] = True
            context.user_data["original_input"] = combined
            # Если среди вопросов есть про проект — покажем кнопки проектов
            kb = _project_keyboard_if_needed(result.clarifying_questions)
            if kb:
                await message.reply_text("Нужны уточнения:\n" + qs, reply_markup=kb)
            else:
                await message.reply_text("Нужны уточнения:\n" + qs + "\n\nПожалуйста, ответьте на вопросы одним сообщением.")
            return
        # Хватает данных: сразу показываем предпросмотр с кнопками
        await _show_preview(message, context, result, combined)
        return

    # Первая попытка извлечения + валидация
    result: ExtractionResult = llm.extract_tasks(text)
    try:
        result = llm.validate_extraction(text, result)
    except Exception:
        pass
    if result.clarifying_questions:
        qs = _format_questions(result.clarifying_questions)
        context.user_data["awaiting_clarifications"] = True
        context.user_data["original_input"] = text
        await message.reply_text("Чтобы корректно поставить задачи, ответьте, пожалуйста, на вопросы:\n" + qs + "\n\nОтветьте одним сообщением.")
        return
    await _show_preview(message, context, result, text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Check access first
    if not await _check_user_access(update, context):
        return
    
    message = update.effective_message
    voice = message.voice
    _touch_activity(context, update.effective_chat.id, update.effective_user.id)
    if context.user_data.get("session_expired"):
        context.user_data.clear()
    if voice is None:
        return

    if not audio.ensure_ffmpeg():
        await message.reply_text("ffmpeg не найден в системе. Установите ffmpeg и попробуйте снова.")
        return

    try:
        file = await context.bot.get_file(voice.file_id)
        ogg_path = await audio.download_to_temp_async(file, suffix=".ogg")
    except Exception:
        logger.exception("Failed to download voice file")
        await message.reply_text("Не удалось скачать голосовое сообщение.")
        return

    ok, wav_path = audio.ogg_to_wav(ogg_path)
    if not ok:
        await message.reply_text("Не удалось конвертировать аудио.")
        return

    model = _ensure_whisper()
    if model is None:
        await message.reply_text("Модель распознавания не загружена.")
        return

    try:
        result = model.transcribe(wav_path)
        transcript = (result.get("text") or "").strip()
    except Exception:
        logger.exception("Transcription failed")
        await message.reply_text("Ошибка распознавания аудио.")
        return

    if not transcript:
        await message.reply_text("Не удалось распознать речь.")
        return

    # Логика аналогична текстовой: сначала пробуем, при необходимости — уточняем
    # Если ожидаем свободные правки голосом
    if context.user_data.get("awaiting_refine_input"):
        pending = context.user_data.get("pending_result")
        original_input = context.user_data.get("original_input", "")
        if not pending:
            context.user_data["awaiting_refine_input"] = False
            await message.reply_text("Нет данных для редактирования. Отправьте новое сообщение.")
            return
        updated = llm.refine_tasks(original_input, pending, transcript)
        try:
            updated = llm.validate_extraction(original_input, updated)
        except Exception:
            pass
        context.user_data["pending_result"] = updated
        context.user_data["awaiting_refine_input"] = False
        await _show_preview(message, context, updated, original_input)
        return

    result: ExtractionResult = llm.extract_tasks(transcript)
    try:
        result = llm.validate_extraction(transcript, result)
    except Exception:
        pass
    if result.clarifying_questions:
        qs = _format_questions(result.clarifying_questions)
        context.user_data["awaiting_clarifications"] = True
        context.user_data["original_input"] = transcript
        await message.reply_text("Нужны уточнения:\n" + qs + "\n\nОтветьте текстом одним сообщением.")
        return
    await _show_preview(message, context, result, transcript)


def _render_preview_text(result: ExtractionResult, original: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    # Check if this is a query-only scenario (no tasks/reminders/updates)
    is_query_only = (
        not result.tasks_new and 
        not result.reminders and 
        not result.tasks_updates and 
        context.user_data.get("query_preview_answer")
    )
    
    # For query-only, just show the answer
    if is_query_only:
        q_prev = context.user_data.get("query_preview_answer")
        return q_prev if q_prev else "(нет ответа)"
    
    # Otherwise, show full preview
    lines = [
        "Исходный текст:",
        original.strip(),
        "",
        "Найденные задачи:",
    ]
    if not result.tasks_new:
        lines.append("— (задач не найдено)")
    for idx, t in enumerate(result.tasks_new, start=1):
        lines += [
            f"#{idx}: {t.title}",
            f"  body: {t.body or '—'}",
            f"  created_at: {t.created_at or '—'}",
            f"  project: {t.project or '—'}",
            f"  labels: {', '.join(t.labels) if t.labels else '—'}",
            f"  priority: {t.priority or '—'}",
            f"  deadline: {_format_local_wall(t.deadline, original_input=original)}",
            f"  direction: {t.direction or '—'}",
            "",
        ]
    # Добавим блок напоминаний (визуализация)
    if result.reminders:
        lines.append("Напоминания (будут созданы):")
        for i, r in enumerate(result.reminders, start=1):
            lines.append(f"{i}. {r.title} (at: {_format_local_wall(r.at, original_input=original) if r.at else '—'}, offset: {r.offset or '—'})")
        lines.append("")

    # Добавим блок планируемых изменений
    if result.tasks_updates:
        lines += ["Планируемые обновления:"]
        mapping = _parse_projects_mapping()
        quick_tasks = []
        if _todoist_enabled():
            try:
                quick_tasks = _get_active_tasks_cached(context)
            except Exception as e:
                logger.warning("Preview: failed to load active tasks for quick match: %s", e)
                quick_tasks = []
        for u in result.tasks_updates:
            target = u.target or ""
            change_items = []
            ch = u.changes
            if ch.title:
                change_items.append(f"title→{ch.title}")
            if ch.description:
                change_items.append("append_description")
            if ch.priority:
                change_items.append(f"priority→{ch.priority}")
            if ch.deadline:
                change_items.append(f"deadline→{_format_local_wall(ch.deadline, original_input=original)}")
            if ch.labels_add:
                change_items.append("labels+=" + ",".join(ch.labels_add))
            if ch.labels_remove:
                change_items.append("labels-=" + ",".join(ch.labels_remove))
            if ch.status:
                change_items.append(f"status→{ch.status}")
            if ch.__dict__.get("project"):
                change_items.append(f"move→{ch.__dict__.get('project')}")
            line = f"- target: {target if target else '(уточнить)'} | changes: {', '.join(change_items) if change_items else '—'}"
            # Быстрая оценка совпадений (если есть токен)
            if quick_tasks:
                matches = _resolve_targets(target, context, mapping, fallback_text=original)
                if matches:
                    sample = ", ".join([m.get('content','') for m in matches[:3]])
                    line += f" | совпадений: {len(matches)} ({sample}{'…' if len(matches)>3 else ''})"
                else:
                    line += " | совпадений: 0"
            lines.append(line)
    # Q&A preview answer if present (and not query-only)
    q_prev = context.user_data.get("query_preview_answer")
    if q_prev and not is_query_only:
        lines += ["Ответ на вопрос:", q_prev]
    return "\n".join(lines).strip()


def _format_questions(questions: list[str]) -> str:
    projects_env = os.getenv("PROJECTS", "").strip()
    # нормализуем список проектов (только имена слева до ":")
    raw = [p.strip() for p in projects_env.split(",") if p.strip()]
    projects = [(x.split(":", 1)[0].strip() if ":" in x else x) for x in raw]
    formatted = []
    for i, q in enumerate(questions, start=1):
        line = f"{i}. {q}"
        q_lower = (q or "").lower()
        if ("проект" in q_lower or "project" in q_lower) and projects:
            line += "\n   Варианты: " + ", ".join(projects)
        formatted.append(line)
    return "\n".join(formatted)


def _parse_projects_mapping() -> dict[str, str]:
    """Parse PROJECTS env as optional Name:ID mapping. Returns {Name: ID}. Names without ID are ignored for mapping."""
    projects_env = os.getenv("PROJECTS", "").strip()
    mapping: dict[str, str] = {}
    for raw in [p.strip() for p in projects_env.split(",") if p.strip()]:
        if ":" in raw:
            name, pid = raw.split(":", 1)
            name = name.strip()
            pid = pid.strip()
            if name and pid:
                mapping[name] = pid
    return mapping


def _todoist_enabled() -> bool:
    return bool(os.getenv("TODOIST_API_TOKEN", "").strip())


def _priority_to_todoist(priority: str | None) -> int | None:
    if not priority:
        return None
    table = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
    return table.get(priority)


def _get_active_tasks_cached(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    ttl = int(os.getenv("ACTIVE_TASKS_TTL_SECONDS", "3600") or "3600")
    store = context.bot_data.setdefault("active_tasks_cache", {})
    now = time.time()
    if store and (now - store.get("ts", 0) < ttl) and store.get("items"):
        return store["items"]
    try:
        items = todoist_client.get_tasks()
        logger.debug(f"Fetched {len(items)} active tasks from Todoist")
    except Exception as e:
        logger.exception("Failed to fetch active tasks from Todoist: %s", e)
        raise
    store["items"] = items
    store["ts"] = now
    return items


def _refresh_active_tasks_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    store = context.bot_data.setdefault("active_tasks_cache", {})
    # Invalidate timestamp to force reload on next call
    store["ts"] = 0
    try:
        _ = _get_active_tasks_cached(context)
    except Exception as e:
        logger.warning("Failed to refresh active tasks cache: %s", e)


def _id_from_url_or_text(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    # try raw id
    if s.isdigit():
        return s
    # try todoist url with id param or path /app/task/<id>
    try:
        u = urlparse(s)
        q = parse_qs(u.query)
        if "id" in q and q["id"]:
            cand = q["id"][0]
            if cand.isdigit():
                return cand
        # path variant: /app/task/<id>
        parts = [p for p in (u.path or "").split("/") if p]
        if len(parts) >= 3 and parts[-2] == "task" and parts[-1].isdigit():
            return parts[-1]
    except Exception:
        pass
    return None


def _resolve_targets(target_text: str, context: ContextTypes.DEFAULT_TYPE, mapping: dict[str, str], *, fallback_text: str | None = None) -> list[dict]:
    """Return list of matched task objects (from active tasks)."""
    # by id/url
    tid = _id_from_url_or_text(target_text)
    if not tid and fallback_text:
        tid = _id_from_url_or_text(fallback_text)
    tasks = _get_active_tasks_cached(context)
    if tid:
        return [t for t in tasks if str(t.get("id")) == str(tid)]
    # by 'last'
    if target_text.strip().lower() in {"last", "последняя", "последний"}:
        last_ids: list[str] = context.bot_data.get("created_task_ids", [])
        if last_ids:
            latest = last_ids[-1]
            return [t for t in tasks if str(t.get("id")) == str(latest)]
    # by text across fields
    q = target_text.lower()
    # build reverse project map id->name from mapping Name:ID
    rev_proj = {v: k for k, v in mapping.items()}
    # basic relative due filters
    user_tz_name = os.getenv("USER_TIMEZONE", "UTC")
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        tz = ZoneInfo(user_tz_name)
    except Exception:
        tz = timezone.utc
    today_local = datetime.now(tz).date()
    due_filter: datetime.date | None = None
    if any(word in q for word in ["сегодня", "today"]):
        due_filter = today_local
    elif any(word in q for word in ["завтра", "tomorrow"]):
        due_filter = today_local + timedelta(days=1)
    results: list[dict] = []
    simple_matches: list[dict] = []
    for t in tasks:
        content = (t.get("content") or "").lower()
        descr = (t.get("description") or "").lower()
        labels = ", ".join(t.get("labels") or []).lower()
        proj_name = rev_proj.get(str(t.get("project_id")), "").lower()
        prio = str(t.get("priority") or "").lower()
        due = t.get("due") or {}
        due_dt = due.get("datetime") or due.get("date") or ""
        due_dt_l = str(due_dt).lower()
        hay = " | ".join([content, descr, labels, proj_name, prio, due_dt_l])
        text_match = (not q) or (q in hay)
        if due_filter and due_dt:
            try:
                if len(due_dt) >= 10:
                    d = due_dt[:10]
                    y, m, d2 = map(int, d.split("-"))
                    if datetime(y, m, d2, tzinfo=timezone.utc).astimezone(tz).date() != due_filter:
                        continue
                else:
                    continue
            except Exception:
                continue
        if text_match:
            simple_matches.append(t)

    # If we have direct/simple matches, prefer them
    if simple_matches:
        results.extend(simple_matches)
    else:
        # Fuzzy matching fallback using rapidfuzz if available
        try:
            from rapidfuzz import fuzz
            min_score = int((os.getenv("MATCH_MIN_SCORE", os.getenv(" MATCH_MIN_SCORE ", "70")) or "70").strip())
            scored: list[tuple[int, dict]] = []
            for t in tasks:
                content = (t.get("content") or "")
                descr = (t.get("description") or "")
                labels = ", ".join(t.get("labels") or [])
                proj_name = rev_proj.get(str(t.get("project_id")), "")
                prio = str(t.get("priority") or "")
                due = t.get("due") or {}
                due_dt = due.get("datetime") or due.get("date") or ""
                hay = " | ".join([content, descr, labels, proj_name, prio, str(due_dt)])
                score = max(
                    fuzz.partial_ratio(q, hay.lower()),
                    fuzz.token_set_ratio(q, hay.lower()),
                )
                if score >= min_score:
                    scored.append((score, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            results.extend([t for _, t in scored])
        except Exception:
            # no fuzzy available; keep results as-is
            pass
    return results


def _project_keyboard_if_needed(questions: list[str]) -> InlineKeyboardMarkup | None:
    # Если среди вопросов есть вопрос про проект, построим клавиатуру из .env
    if not any((q or "").lower().find("проект") != -1 or (q or "").lower().find("project") != -1 for q in questions):
        return None
    projects_env = os.getenv("PROJECTS", "").strip()
    raw = [p.strip() for p in projects_env.split(",") if p.strip()]
    projects = [(x.split(":", 1)[0].strip() if ":" in x else x) for x in raw]
    if not projects:
        return None
    rows = []
    for p in projects:
        rows.append([InlineKeyboardButton(p, callback_data=f"clarify:project:{p}")])
    rows.append([InlineKeyboardButton("Пропустить", callback_data="clarify:project:")])
    return InlineKeyboardMarkup(rows)


def _build_preview_kb(result: ExtractionResult) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("✅ Подтвердить и отправить JSON", callback_data="preview:confirm")],
        [InlineKeyboardButton("✏️ Изменить", callback_data="preview:edit"), InlineKeyboardButton("❌ Отмена", callback_data="preview:cancel")],
    ]
    return InlineKeyboardMarkup(buttons)


def _intent_list(result: ExtractionResult) -> List[Dict[str, Any]]:
    try:
        meta = getattr(result, "meta", None)
        intents = getattr(meta, "intents", None)
        return intents or []
    except Exception:
        return []


def _extract_query_intent(result: ExtractionResult) -> Optional[Dict[str, Any]]:
    for it in _intent_list(result):
        if str(it.get("type", "")).lower() == "query":
            return it
    return None


def _server_filter_from_query(it: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Build server-side filter params for Todoist where possible.
    Returns dict with keys: filter, project_id, label
    """
    filters = it.get("filters") or {}
    time_f = (filters.get("time") or {}) if isinstance(filters.get("time"), dict) else {}
    due = (time_f.get("due") or "").lower()
    filter_parts: List[str] = []
    if due in {"today", "tomorrow", "overdue", "this_week", "next_week"}:
        # Todoist filter language keywords
        if due == "this_week":
            filter_parts.append("today | overdue | 7 days")
        elif due == "next_week":
            filter_parts.append("next 7 days")
        else:
            filter_parts.append(due)
    due_on = (time_f.get("due_on") or "").strip()
    if due_on:
        filter_parts.append(f"due on: {due_on}")
    # project
    proj = (filters.get("project") or {})
    proj_names = proj.get("names") or []
    project_id = None
    if proj_names:
        name = str(proj_names[0]).strip()
        project_id = mapping.get(name)
    # labels
    labels_f = (filters.get("labels") or {})
    include = labels_f.get("include") or []
    label = None
    if include:
        label = str(include[0]).strip()
        # server filter also can accept @label in filter string, but we use label param for simplicity
    # Assemble filter string
    filter_str = " & ".join([p for p in filter_parts if p]) or None
    return {"filter": filter_str, "project_id": project_id, "label": label}


def _apply_local_filters(tasks: List[Dict[str, Any]], it: Dict[str, Any]) -> List[Dict[str, Any]]:
    filters = it.get("filters") or {}
    # priority filter
    pr = (filters.get("priority") or {})
    allowed = set((pr.get("in") or []))
    if allowed:
        def _ok_p(t):
            return str(t.get("priority") or "").lower() in allowed
        tasks = [t for t in tasks if _ok_p(t)]
    # text contains
    tx = (filters.get("text") or {})
    contains = [str(x).lower() for x in (tx.get("contains") or [])]
    if contains:
        def _ok_text(t):
            hay = ((t.get("content") or "") + " " + (t.get("description") or "")).lower()
            return all(x in hay for x in contains)
        tasks = [t for t in tasks if _ok_text(t)]
    return tasks


def _maybe_prepare_query_preview(context: ContextTypes.DEFAULT_TYPE, result: ExtractionResult) -> None:
    it = _extract_query_intent(result)
    if not it:
        context.user_data["query_preview_answer"] = ""
        return
    mapping = _parse_projects_mapping()
    params = _server_filter_from_query(it, mapping)
    try:
        tasks = todoist_client.get_tasks(
            filter=params.get("filter"),
            project_id=params.get("project_id"),
            label=params.get("label"),
        )
    except Exception as e:
        logger.warning("Query server fetch failed: %s", e)
        tasks = []
    tasks = _apply_local_filters(tasks, it)
    # Build reverse project map id->name for answer rendering
    rev = {v: k for k, v in mapping.items()}
    tz = os.getenv("USER_TIMEZONE", "UTC") or "UTC"
    question = str(it.get("question") or "Вопрос о задачах").strip()
    try:
        draft = llm.answer_about_tasks(question, tasks, rev, tz)
        final = llm.validate_answer(question, tasks, draft) if draft else ""
    except Exception:
        final = ""
    context.user_data["query_preview_answer"] = final or ""


async def _show_preview(message, context: ContextTypes.DEFAULT_TYPE, result: ExtractionResult, original_input: str):
    # Check if this is a query-only scenario (no tasks/reminders/updates)
    is_query_only = (
        not result.tasks_new and 
        not result.reminders and 
        not result.tasks_updates and 
        context.user_data.get("query_preview_answer")
    )
    
    # For query-only, just send the answer directly without preview buttons
    if is_query_only:
        answer = context.user_data.get("query_preview_answer", "")
        if answer:
            await message.reply_text(answer)
        return
    
    # Otherwise, show preview with buttons
    context.user_data["pending_result"] = result
    context.user_data["original_input"] = original_input
    # Подготовим превью ответа на query (если есть)
    try:
        _maybe_prepare_query_preview(context, result)
    except Exception:
        logger.debug("Query preview preparation failed", exc_info=True)
    preview = _render_preview_text(result, original_input, context)
    await message.reply_text(preview, reply_markup=_build_preview_kb(result))


def _apply_edit_from_text(result: ExtractionResult, user_text: str) -> ExtractionResult:
    # Простой парсер: строки вида
    # задача: 1\nполе: priority\nзначение: high
    text = user_text.strip()
    lower = text.lower()
    def _extract(prefix: str) -> str | None:
        for line in text.splitlines():
            if line.lower().startswith(prefix):
                return line.split(":", 1)[1].strip()
        return None
    idx_str = _extract("задача") or _extract("task")
    field = _extract("поле") or _extract("field")
    value = _extract("значение") or _extract("value")
    try:
        idx = int((idx_str or "0").strip()) - 1
    except Exception:
        idx = -1
    if idx < 0 or idx >= len(result.tasks_new) or not field:
        return result
    task = result.tasks_new[idx]
    # Нормализация простых полей
    if field in {"title", "body", "created_at", "project", "priority", "deadline", "direction"}:
        setattr(task, field, value)
    elif field in {"labels"}:
        labels = [p.strip() for p in (value or "").split(",") if p.strip()] if value else []
        task.labels = labels
    # Пересобираем объект
    result.tasks_new[idx] = task
    return result


async def on_preview_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check access first
    if not await _check_user_access(update, context):
        return
    
    q = update.callback_query
    if not q:
        return
    await q.answer()
    # Активность по кнопке — обновляем таймер, снимаем флаг истечения
    _touch_activity(context, q.message.chat.id, q.from_user.id)
    if context.user_data.get("session_expired"):
        context.user_data["session_expired"] = False
    data = q.data or ""
    chat = q.message
    pending: ExtractionResult | None = context.user_data.get("pending_result")
    if data == "preview:cancel":
        context.user_data["pending_result"] = None
        context.user_data["awaiting_edit_input"] = False
        await chat.reply_text("Отменено. Отправьте новое сообщение, чтобы начать заново.")
        try:
            await q.message.delete()
        except Exception:
            pass
        return
    if data == "preview:edit":
        context.user_data["awaiting_refine_input"] = True
        await chat.reply_text(
            "Опишите, что нужно изменить, любым текстом или пришлите голосовое.\n"
            "Например: 'в задаче 1 поменяй дедлайн на завтра 14:00 и приоритет high; в задаче 2 проект на Работа'.\n"
            "Я применю правки и покажу обновлённый предпросмотр."
        )
        return
    if data == "preview:confirm":
        if not pending:
            await chat.reply_text("Нет данных для подтверждения.")
            return
        # интеграция с Todoist (если задан токен)
        if _todoist_enabled():
            mapping = _parse_projects_mapping()
            created_infos: list[str] = []
            local_created_ids: list[str] = []  # IDs задач, созданных в этом запросе

            for t in pending.tasks_new:
                # описание: body + source_text (если есть)
                desc_parts = []
                if t.body:
                    desc_parts.append(t.body)
                if t.source_text:
                    desc_parts.append(f"Исходный текст: {t.source_text}")
                description = "\n\n".join(desc_parts) if desc_parts else None
                # Project is disabled - all tasks go to Inbox
                project_id = None
                # метки: то, что прислал ИИ, плюс direction как отдельная метка, если задан
                labels_set = {(lab or "").strip().lower() for lab in (t.labels or []) if (lab or "").strip()}
                if t.direction:
                    labels_set.add(str(t.direction).strip().lower())
                labels = list(labels_set)
                # приоритет
                prio = _priority_to_todoist(t.priority)
                # дедлайн: как есть (ISO UTC)
                # дедлайн: отправляем локальное время с явным смещением TZ пользователя
                if t.deadline:
                    _force_local = _should_force_local_from_input(context.user_data.get("original_input", ""))
                    due_dt = _to_local_with_offset(t.deadline, force_local=_force_local)
                else:
                    due_dt = None
                try:
                    resp = todoist_client.create_task(
                        content=t.title,
                        description=description,
                        project_id=project_id,
                        labels=labels or None,
                        priority=prio,
                        due_datetime=due_dt,
                    )
                    url = resp.get("url") or ""
                    tid = resp.get("id") or ""
                    if tid:
                        local_created_ids.append(str(tid))
                    # store last created ids (buffer 20)
                    buf: list[str] = context.bot_data.setdefault("created_task_ids", [])
                    buf.append(str(tid))
                    if len(buf) > 20:
                        del buf[: len(buf) - 20]
                    created_infos.append(f"✅ Создано в Todoist: {t.title} ({tid}) {url}")
                except Exception as e:
                    created_infos.append(f"❌ Не удалось создать: {t.title} — {e}")

            # --- НАПОМИНАНИЯ ---
            if pending.reminders:
                # Если есть новые задачи, привязываем к первой
                target_task_id = local_created_ids[0] if local_created_ids else None
                
                for r in pending.reminders:
                    # Если нет целевой задачи, сначала попробуем найти существующую
                    if not target_task_id:
                        # Пытаемся найти задачу по названию напоминания
                        original_input = context.user_data.get("original_input", "")
                        try:
                            matches = _resolve_targets(r.title, context, mapping, fallback_text=original_input)
                            if matches:
                                target_task_id = str(matches[0].get("id"))
                                created_infos.append(f"💡 Напоминание привязано к существующей задаче: {matches[0].get('content')}")
                        except Exception:
                            pass
                    
                    # Если не нашли и не создали - создаем задачу-пустышку под напоминание
                    if not target_task_id:
                        try:
                            # Создаем задачу с названием напоминания
                            t_resp = todoist_client.create_task(content=f"Напоминание: {r.title}")
                            target_task_id = t_resp.get("id")
                            t_url = t_resp.get("url")
                            created_infos.append(f"✅ Создана задача для напоминания: {r.title} {t_url}")
                        except Exception as e:
                            created_infos.append(f"❌ Не удалось создать задачу для напоминания: {r.title} — {e}")
                            continue
                    
                    # Создаем напоминание
                    if target_task_id:
                        try:
                            # Формируем due object
                            due_obj = {}
                            if r.at:
                                # r.at is ISO UTC string. Todoist wants specific format or string.
                                # Проще всего передать как string, если API поддерживает, или сконвертировать.
                                # create_reminder doc says: due object example: {"string": "tomorrow at 10:00"} or {"datetime": "..."}
                                # Передадим datetime в ISO формате (Todoist принимает)
                                due_obj["date"] = r.at # Попробуем поле date (для ISO8601) или datetime?
                                # API v2: due_date (YYYY-MM-DD) or due_string.
                                # Но для напоминаний (reminders endpoint):
                                # "due": { "date": "...", "datetime": "...", "string": "..." }
                                # Используем datetime, но нужно убедиться, что он в правильном формате.
                                # r.at у нас YYYY-MM-DDTHH:MM:SSZ.
                                # Попробуем передать как "string" для надежности парсинга Todoist'ом, если это natural language?
                                # Нет, r.at это уже ISO. Передадим как "datetime" (но нужно убрать Z если Todoist не любит, или оставить).
                                # Безопаснее всего сконвертировать в локальное время и передать как string?
                                # Или просто передать r.at как "datetime".
                                # Попробуем "datetime": r.at
                                due_obj = {"datetime": r.at}
                            else:
                                # Если времени нет, напоминание не имеет смысла без времени?
                                # Или offset?
                                # Если offset задан (PT15M), то это relative. Но API напоминаний обычно требует конкретного времени.
                                # Если offset, то нужно прибавить к now? Или к due date задачи?
                                # Пока поддержим только явное время r.at.
                                pass
                            
                            if due_obj:
                                todoist_client.create_reminder(item_id=target_task_id, due=due_obj)
                                created_infos.append(f"⏰ Напоминание установлено: {r.title} на {r.at}")
                            else:
                                created_infos.append(f"⚠️ Напоминание пропущено (нет времени): {r.title}")

                        except Exception as e:
                            created_infos.append(f"❌ Ошибка установки напоминания: {r.title} — {e}")

            if created_infos:
                # Personalize the confirmation message
                base_msg = "\n".join(created_infos)
                try:
                    personalized_msg = llm.personalize_message(
                        base_msg,
                        context=f"Task creation confirmation. {len(created_infos)} item(s) created."
                    )
                    await chat.reply_text(personalized_msg)
                except Exception:
                    # Fallback to base message
                    await chat.reply_text(base_msg)
            # Force refresh cache so text-based updates can see newly created tasks
            try:
                _refresh_active_tasks_cache(context)
            except Exception:
                pass

            # process updates
            update_infos: list[str] = []
            max_auto = int(os.getenv("MAX_AUTO_APPLY_MATCHES", "10") or "10")
            for upd in (pending.tasks_updates or []):
                targets = _resolve_targets(upd.target or "", context, mapping, fallback_text=context.user_data.get("original_input", ""))
                if not targets:
                    update_infos.append(f"⚠️ Не найдено задач по запросу: {upd.target}")
                    continue
                if len(targets) > max_auto:
                    update_infos.append(f"⚠️ Слишком много совпадений ({len(targets)}). Уточните запрос для: {upd.target}")
                    continue
                for task in targets:
                    tid = str(task.get("id"))
                    ch = upd.changes
                    current_desc = (task.get("description") or "").strip()
                    current_labels = task.get("labels") or []

                    # build new values
                    new_title = ch.title if hasattr(ch, "title") and ch.title else None
                    add_descr = ch.description if hasattr(ch, "description") and ch.description else None
                    new_labels = list(current_labels)
                    if ch.labels_add:
                        new_labels.extend([l.strip().lower() for l in ch.labels_add if l.strip()])
                    if ch.labels_remove:
                        remove_set = {l.strip().lower() for l in ch.labels_remove if l.strip()}
                        new_labels = [l for l in new_labels if l.lower() not in remove_set]
                    new_labels = list(dict.fromkeys(new_labels)) if new_labels else None

                    prio = _priority_to_todoist(ch.priority) if hasattr(ch, "priority") and ch.priority else None
                    due_dt = None
                    if hasattr(ch, "deadline") and ch.deadline:
                        force_local = _should_force_local_from_input(context.user_data.get("original_input", ""))
                        due_dt = _to_local_with_offset(ch.deadline, force_local=force_local)

                    notes_parts: list[str] = []
                    # status
                    status = ch.status if hasattr(ch, "status") else None
                    # project move
                    proj_name = getattr(ch, "project", None)
                    move_pid = mapping.get(proj_name or "", None)

                    if new_title:
                        notes_parts.append(f"title→{new_title}")
                    if prio is not None:
                        notes_parts.append(f"priority→{prio}")
                    if due_dt:
                        notes_parts.append(f"deadline→{due_dt}")
                    if add_descr:
                        notes_parts.append(f"append_description")
                    if ch.labels_add:
                        notes_parts.append(f"labels+={','.join(ch.labels_add)}")
                    if ch.labels_remove:
                        notes_parts.append(f"labels-={','.join(ch.labels_remove)}")
                    if move_pid:
                        notes_parts.append(f"move→{proj_name}")
                    if status:
                        notes_parts.append(f"status→{status}")

                    # Format description update with timestamp
                    new_description = current_desc
                    if add_descr:
                        # Get current timestamp in user's timezone
                        try:
                            from zoneinfo import ZoneInfo
                            tzname = os.getenv("USER_TIMEZONE", "UTC") or "UTC"
                            tz = ZoneInfo(tzname)
                        except Exception:
                            tz = timezone.utc
                        now_local = datetime.now(tz)
                        timestamp = now_local.strftime("%Y-%m-%d %H:%M")
                        upd_text = f"UPD {timestamp}: {add_descr}"
                        new_description = (new_description + ("\n\n" if new_description else "") + upd_text).strip()

                    # apply field updates
                    try:
                        if any(v is not None for v in [new_title, new_description, new_labels, prio, due_dt]):
                            todoist_client.update_task(
                                tid,
                                content=new_title,
                                description=new_description,
                                labels=new_labels,
                                priority=prio,
                                due_datetime=due_dt,
                            )
                        if move_pid:
                            todoist_client.move_task(tid, project_id=move_pid)
                        if status == "done":
                            todoist_client.close_task(tid)
                        elif status == "todo":
                            todoist_client.reopen_task(tid)
                        task_url = f"https://app.todoist.com/app/task/{tid}"
                        update_infos.append(f"✅ Обновлено: {task.get('content')} ({tid}) {task_url}")
                    except Exception as e:
                        update_infos.append(f"❌ Ошибка обновления {task.get('content')} ({tid}) — {e}")
            if update_infos:
                # Personalize the update confirmation message
                base_msg = "\n".join(update_infos)
                try:
                    personalized_msg = llm.personalize_message(
                        base_msg,
                        context=f"Task update confirmation. {len(update_infos)} item(s) updated."
                    )
                    await chat.reply_text(personalized_msg)
                except Exception:
                    # Fallback to base message
                    await chat.reply_text(base_msg)
        # Если в превью был ответ по вопросу (query) — отправим его пользователю
        try:
            q_answer: str = context.user_data.get("query_preview_answer", "")
            if q_answer:
                await chat.reply_text(q_answer)
        except Exception:
            pass
        # очистим состояние
        context.user_data["pending_result"] = None
        context.user_data["awaiting_edit_input"] = False
        context.user_data["original_input"] = ""
        # удалим сообщение с кнопками-предпросмотра
        try:
            await q.message.delete()
        except Exception:
            pass
        return


async def on_clarify_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    await q.answer()
    # продлеваем таймер
    _touch_activity(context, q.message.chat.id, q.from_user.id)
    data = q.data or ""
    if not data.startswith("clarify:"):
        return
    # сейчас поддерживаем только выбор проекта
    if data.startswith("clarify:project:"):
        chosen = data.split(":", 2)[2]
        original_input: str = context.user_data.get("original_input", "")
        if not original_input:
            await q.message.reply_text("Нет исходного текста для уточнения. Отправьте новое сообщение.")
            return
        # добавляем уточнение и переизвлекаем
        extra = f"\n\nУточнение пользователя: проект={chosen or 'null'}\n"
        combined = original_input + extra
        result: ExtractionResult = llm.extract_tasks(combined)
        # сбрасываем ожидание уточнений, но сохраняем возможность новых
        context.user_data["awaiting_clarifications"] = False
        context.user_data["original_input"] = ""
        if result.clarifying_questions:
            qs = _format_questions(result.clarifying_questions)
            kb = _project_keyboard_if_needed(result.clarifying_questions)
            context.user_data["awaiting_clarifications"] = True
            context.user_data["original_input"] = combined
            if kb:
                await q.message.reply_text("Нужны уточнения:\n" + qs, reply_markup=kb)
            else:
                await q.message.reply_text("Нужны уточнения:\n" + qs + "\n\нОтветьте одним сообщением.")
            return
        await _show_preview(q.message, context, result, combined)

def _touch_activity(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> None:
    # отменяем предыдущий таймер
    job = context.user_data.get("timeout_job")
    if job:
        try:
            job.schedule_removal()
        except Exception:
            pass
    # если job_queue ещё не инициализирован, выходим без планирования
    jq = getattr(context, "job_queue", None)
    if jq is None:
        return
    # ставим новый
    job = jq.run_once(
        _on_session_timeout,
        when=SESSION_TIMEOUT_SECONDS,
        data={"chat_id": chat_id, "user_id": user_id},
        name=f"timeout:{chat_id}:{user_id}",
    )
    context.user_data["timeout_job"] = job


async def _on_session_timeout(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    job = ctx.job
    data = getattr(job, "data", {}) or {}
    chat_id = data.get("chat_id")
    # Помечаем сессию как истекшую: сбрасываем ожидания, но сохраняем pending_result для возможного продолжения
    # user_data доступ к контексту чата через application.chat_data? Используем per-user user_data через ctx.application.
    # В PTB 21 в колбэке job нет прямого user_data; используем application.user_data[(chat_id, user_id)].
    user_id = data.get("user_id")
    try:
        ud = ctx.application.user_data.get(user_id)
        if isinstance(ud, dict):
            ud["awaiting_clarifications"] = False
            ud["awaiting_refine_input"] = False
            ud["session_expired"] = True
            ud["timeout_job"] = None
    except Exception:
        pass
    try:
        await ctx.application.bot.send_message(
            chat_id=chat_id,
            text=(
                "Сессия завершена из-за неактивности (3 минуты). Вы можете продолжить с места остановки, нажав кнопки в последнем сообщении, "
                "или отправьте новое сообщение/голосовое, чтобы начать заново."
            ),
        )
    except Exception:
        pass


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    app.add_handler(CallbackQueryHandler(on_preview_callback, pattern=r"^preview:"))
    app.add_handler(CallbackQueryHandler(on_clarify_callback, pattern=r"^clarify:"))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Bot started")
    # Синхронный вызов, сам управляет циклом событий
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
