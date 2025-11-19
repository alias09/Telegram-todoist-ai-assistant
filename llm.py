from __future__ import annotations

import os
import re
import httpx
import orjson
from typing import Dict, Any, List, Optional
from schema import ExtractionResult

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT_BASE = (
    "You are a task extraction engine. You will receive a user message in Russian or English "
    "describing tasks, updates, and reminders. The message may be written in third-person, forwarded, or addressed to someone else (e.g., 'Валера, купи ...'). You must still extract tasks. "
    "Return ONLY a single JSON object, matching exactly the provided schema. No markdown, no comments, no extra text. "
    "All dates must be ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ). If something is not present, use null or empty arrays. "
    "Always include the original input text in 'source_text' fields where appropriate. "
    "If required information to create/update tasks is missing, generate concise clarifying questions in 'clarifying_questions' (Russian if the input is Russian). "
    "Additionally, ALWAYS fill meta.intents as an ordered list describing user intents among: clarify, create, update, move, status, reminder, query. "
    "For query intents, fill meta.intents[*].filters with your best guess of filters (time/project/labels/priority/text) and include a natural language 'question'. "
    "IMPORTANT: The user may provide MULTIPLE distinct requests in one message (e.g. 'Create task A AND update task B'). You must extract ALL of them into the respective arrays."
)

USER_INSTRUCTIONS = (
    "Extract the following fields into the JSON schema: tasks_new, tasks_updates, reminders, clarifying_questions, meta. "
    "Always include meta.intents as an ordered array of objects. Each item MUST have 'type' among: clarify, create, update, move, status, reminder, query. "
    "Prefer a single unified JSON object (do not return multiple JSONs). Use meta.intents[*].refs to reference indices of tasks_new / tasks_updates, when applicable. "
    "\n\n"
    "CRITICAL: Distinguish between QUERY (find/show existing tasks) and CREATE (add new tasks):\n"
    "- QUERY intent: user asks to FIND, SHOW, LIST, or GET information about EXISTING tasks (e.g., 'найди задачу', 'покажи задачи', 'какие задачи', 'есть ли задача'). Use intent type 'query' and DO NOT fill tasks_new.\n"
    "- CREATE intent: user explicitly asks to CREATE, ADD, or SET a NEW task (e.g., 'создай задачу', 'добавь задачу', 'поставь задачу', 'напомни мне'). Use intent type 'create' and fill tasks_new.\n"
    "\n"
    "For query intents, include meta.intents[*].filters object with optional keys: time.due ('today'|'tomorrow'|'overdue'|'this_week'|'next_week' or null), time.due_on ('YYYY-MM-DD'), time.due_between {from,to}, project.names [string], labels.include/exclude [string], priority.in ['low','medium','high','urgent'], text.contains [string], status ('active'). Also include a human 'question'. "
    "\n"
    "If the user asks to MODIFY EXISTING tasks (phrases like: 'изменить', 'поменять', 'добавь/добавить' [к существующей], 'исправь', 'обнови', 'пополни', 'допиши'), then FILL tasks_updates. "
    "If the user asks to CREATE NEW tasks, FILL tasks_new. "
    "If the user asks for REMINDERS (e.g. 'напомни мне', 'remind me'), FILL reminders. "
    "HANDLE MIXED INTENTS: If the user says 'Create X and Update Y', you MUST fill both tasks_new AND tasks_updates. "
    "For updates that add info, put that text into changes.description (to be appended to the task description). "
    "For tasks_new use the FIXED FORMAT fields: title, body, created_at, project, labels, priority, deadline, direction, source_text. "
    "'project' MUST ALWAYS be null - do not ask clarifying questions about projects. "
    "'direction' is 'from_me' (я ставлю) or 'to_me' (мне ставят). RULES: "
    "(1) If the speaker (first-person: я/мне/меня/мой/моя/мои) is the executor (e.g., 'поставь задачу, чтобы я ...', 'создай задачу, чтобы мне ...', 'напомни мне ...', 'я должен ...'), set direction='to_me' EVEN IF the text uses an imperative addressed to the assistant. "
    "(2) If the text clearly assigns a task to a third person by name or role (imperative + explicit addressee: 'Валера, ...', 'Саше ...', 'Команде ...'), set direction='from_me'. "
    "(3) If both the speaker and a third person are mentioned or direction is unclear, leave direction=null and add a concise clarifying question like 'Эта задача для вас или для Валеры?'. "
    "When still ambiguous after applying (1)-(3), default to 'to_me'. "
    "LABELS: infer 1–5 concise, meaningful labels per task from context even if not explicitly provided. Use domain-relevant keywords, avoid junk, avoid duplicates, lowercase. If no sensible labels, use an empty list. "
    "Infer language in meta.language. If uncertain, set meta.confidence conservatively. "
)

SCHEMA_HINT = {
    "tasks_new": [
        {
            "title": "string",
            "body": "string|null",
            "created_at": "YYYY-MM-DDTHH:MM:SSZ",
            "project": "Дом|Работа|Личное|null",
            "labels": ["string"],
            "priority": "low|medium|high|urgent|null",
            "deadline": "YYYY-MM-DDTHH:MM:SSZ|null",
            "direction": "from_me|to_me|null",
            "source_text": "string"
        }
    ],
    "tasks_updates": [
        {
            "target": "task_id_or_title",
            "changes": {
                "status": "todo|in_progress|blocked|done|null",
                "title": "string|null",
                "description": "string|null",
                "priority": "low|medium|high|urgent|null",
                "labels_add": ["string"],
                "labels_remove": ["string"],
                "assignee": "string|null",
                "deadline": "YYYY-MM-DDTHH:MM:SSZ|null"
            },
            "source_text": "string"
        }
    ],
    "reminders": [
        {
            "title": "string",
            "at": "YYYY-MM-DDTHH:MM:SSZ|null",
            "offset": "PT15M|null",
            "repeat": "none|daily|weekly|monthly|null",
            "source_text": "string"
        }
    ],
    "clarifying_questions": ["string"],
    "meta": {
        "language": "ru|en|…",
        "parsed_at": "YYYY-MM-DDTHH:MM:SSZ",
        "confidence": 0.0
    }
}


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\\n", "", text)
        text = text[:-3]
    return text


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return orjson.loads(text)
    except Exception:
        return {}


def _build_system_prompt() -> str:
    role = os.getenv("AGENT_ROLE", "").strip()
    user_name = os.getenv("USER_NAME", "").strip()
    user_role = os.getenv("USER_ROLE", "").strip()
    user_timezone = os.getenv("USER_TIMEZONE", "").strip()
    user_profile_json = os.getenv("USER_PROFILE", "").strip()
    parts: List[str] = []
    if role:
        parts.append(f"AGENT ROLE: {role}")
    profile_lines: List[str] = []
    if user_name:
        profile_lines.append(f"name={user_name}")
    if user_role:
        profile_lines.append(f"role={user_role}")
    if user_timezone:
        profile_lines.append(f"timezone={user_timezone}")
    if user_profile_json:
        profile_lines.append(f"extra={user_profile_json}")
    if profile_lines:
        parts.append("USER PROFILE: " + "; ".join(profile_lines))
    if parts:
        return SYSTEM_PROMPT_BASE + "\n\n" + "\n".join(parts)
    return SYSTEM_PROMPT_BASE


def _get_models_list() -> List[str]:
    models_env = os.getenv("OPENROUTER_MODEL", "").strip()
    if not models_env:
        return ["anthropic/claude-3.5-sonnet"]
    parts = [m.strip() for m in models_env.split(",") if m.strip()]
    return parts or ["anthropic/claude-3.5-sonnet"]


def _call_openrouter(model: str, messages: list[dict], temperature: float = 0.1, extra_headers: Dict[str, str] | None = None) -> Dict[str, Any] | None:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    app_url = os.getenv("OPENROUTER_APP_URL", "").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-Title"] = app_title
    if extra_headers:
        headers.update(extra_headers)
    payload = {"model": model, "messages": messages, "temperature": temperature}
    with httpx.Client(timeout=60) as client:
        r = client.post(OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


def extract_tasks(text: str) -> ExtractionResult:
    # Projects are disabled - all tasks go to Inbox
    schema_hint = SCHEMA_HINT.copy()
    schema_hint["tasks_new"][0]["project"] = "null"  # Always null
    
    examples = (
        "EXAMPLES (for direction):\n"
        "A) 'Поставь задачу, чтобы я купил хлеб' -> direction: to_me.\n"
        "B) 'Валера, купи хлеб' -> direction: from_me.\n"
        "C) 'Поставь задачу Валере, чтобы он купил хлеб' -> direction: from_me; if unclear, ask who should do it.\n"
        "\nEXAMPLES (for updates):\n"
        "Input: 'Надо изменить задачу по покупке хлеба, добавить, что нужен именно черный хлеб' -> tasks_updates: [{target: 'покупке хлеба', changes: {description: 'нужен именно черный хлеб'}}]; tasks_new: []\n"
        "Input: 'В задаче найти поставщиков по работе подними приоритет до high и поставь дедлайн завтра 14:00' -> tasks_updates: [{target: 'найти поставщиков по работе', changes: {priority: 'high', deadline: '<завтра 14:00 в ISO>'}}]\n"
    )
    user_content = (
        USER_INSTRUCTIONS
        + "\n\nIMPORTANT: project field must ALWAYS be null. Do not ask about projects."
        + "\n\nSCHEMA: "
        + orjson.dumps(schema_hint).decode("utf-8")
        + "\n\n" + examples
        + "\n\nTEXT:\n"
        + text
    )
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    for m in _get_models_list():
        try:
            data = _call_openrouter(m, messages)
            if not data:
                continue
            content = data["choices"][0]["message"]["content"]
            content = _strip_code_fences(content)
            obj = _safe_json_loads(content)
            return ExtractionResult.model_validate(obj)
        except Exception:
            continue
    return ExtractionResult()


def refine_tasks(original_input: str, current: ExtractionResult, corrections_text: str) -> ExtractionResult:
    import logging
    logger = logging.getLogger(__name__)
    
    refine_instructions = (
        "You will be given: (1) the original user input, (2) the CURRENT JSON extraction, and (3) the user's FREE-FORM corrections.\n"
        "CRITICAL: Apply ALL requested changes from the corrections. Do not skip any changes mentioned by the user.\n"
        "- If user mentions deadline/time - update the deadline field\n"
        "- If user mentions additional information/details/description - append it to the body field\n"
        "- If user mentions both - update BOTH fields\n"
        "Preserve unspecified fields. Keep the same schema as before (tasks_new, tasks_updates, reminders, clarifying_questions, meta).\n"
        "Return ONLY the FULL corrected JSON object. No prose. No markdown."
    )
    user_payload = (
        "REFINE INSTRUCTIONS:\n" + refine_instructions +
        "\n\nORIGINAL INPUT:\n" + original_input +
        "\n\nCURRENT JSON:\n" + orjson.dumps(current.model_dump(mode="python")).decode("utf-8") +
        "\n\nUSER CORRECTIONS:\n" + corrections_text + "\n"
    )
    
    logger.info(f"refine_tasks called with corrections: {corrections_text}")
    logger.debug(f"Current JSON before refinement: {orjson.dumps(current.model_dump(mode='python')).decode('utf-8')}")
    
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_payload},
    ]
    for m in _get_models_list():
        try:
            data = _call_openrouter(m, messages)
            if not data:
                continue
            content = data["choices"][0]["message"]["content"]
            content = _strip_code_fences(content)
            obj = _safe_json_loads(content)
            result = ExtractionResult.model_validate(obj)
            logger.info(f"refine_tasks result: {orjson.dumps(result.model_dump(mode='python')).decode('utf-8')}")
            return result
        except Exception as e:
            logger.warning(f"refine_tasks failed with model {m}: {e}")
            continue
    logger.warning("refine_tasks: all models failed, returning current")
    return current


def _get_validator_model() -> Optional[str]:
    model = os.getenv("OPENROUTER_VALIDATOR_MODEL", "").strip()
    return model or None


def _validator_enabled() -> bool:
    val = os.getenv("VALIDATOR_ENABLED", "true").strip().lower()
    return val in {"1", "true", "yes", "on"}


def validate_extraction(original_input: str, candidate: ExtractionResult) -> ExtractionResult:
    """Use a secondary model to validate and if needed minimally correct the extracted JSON."""
    if not _validator_enabled():
        return candidate
    model = _get_validator_model()
    if not model:
        return candidate
    try:
        user_payload = (
            "You are a strict validator. Given ORIGINAL INPUT and CANDIDATE JSON (matching schema), "
            "check for faithfulness, missing required fields, obvious inconsistencies (dates/priority/project), and return ONLY corrected full JSON.\n\n"
            "ORIGINAL INPUT:\n" + original_input + "\n\nCANDIDATE JSON:\n" + orjson.dumps(candidate.model_dump(mode="python")).decode("utf-8")
        )
        messages = [
            {"role": "system", "content": "Return ONLY full corrected JSON object. No prose."},
            {"role": "user", "content": user_payload},
        ]
        data = _call_openrouter(model, messages, temperature=0.0)
        if not data:
            return candidate
        content = data["choices"][0]["message"]["content"]
        content = _strip_code_fences(content)
        obj = _safe_json_loads(content)
        return ExtractionResult.model_validate(obj)
    except Exception:
        return candidate


def answer_about_tasks(question: str, tasks: List[Dict[str, Any]], projects_map: Dict[str, str], timezone: str) -> str:
    """Generate a concise natural-language answer about user's tasks based on provided context."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ""
    # Compact the context
    def _shorten(s: str, n: int = 200) -> str:
        s = s or ""
        return s if len(s) <= n else s[: n - 1] + "…"
    ctx_lines: List[str] = []
    for t in tasks[:500]:  # hard cap
        proj_name = projects_map.get(str(t.get("project_id")), "")
        due = t.get("due") or {}
        due_dt = due.get("datetime") or due.get("date") or ""
        labels = ", ".join(t.get("labels") or [])
        ctx_lines.append(
            f"- id={t.get('id')} | project={proj_name} | prio={t.get('priority')} | labels=[{labels}] | due={due_dt} | title={_shorten(t.get('content') or '', 160)} | desc={_shorten(t.get('description') or '', 120)}"
        )
    ctx_text = "\n".join(ctx_lines)
    prompt = (
        "You are an assistant answering questions about a user's Todoist tasks. "
        "Use ONLY the provided TASKS CONTEXT. If uncertain, say so briefly. Return a concise answer in the input language.\n\n"
        f"USER TIMEZONE: {timezone or 'UTC'}\n\nQUESTION:\n{question}\n\nTASKS CONTEXT (active tasks):\n{ctx_text}\n"
    )
    messages = [
        {"role": "system", "content": "Answer concisely in Russian if the question is Russian. No markdown."},
        {"role": "user", "content": prompt},
    ]
    for m in _get_models_list():
        try:
            data = _call_openrouter(m, messages)
            if not data:
                continue
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return ""


def validate_answer(question: str, tasks: List[Dict[str, Any]], draft_answer: str) -> str:
    """Validate/correct a natural language answer via secondary model."""
    if not _validator_enabled():
        return draft_answer
    model = _get_validator_model()
    if not model:
        return draft_answer
    try:
        ctx_lines = []
        for t in tasks[:300]:
            ctx_lines.append(orjson.dumps(t).decode("utf-8"))
        ctx_text = "\n".join(ctx_lines)
        user_payload = (
            "Given QUESTION, TASKS JSON CONTEXT and DRAFT ANSWER, verify factual correctness and adjust numbers/lists if needed. "
            "Return ONLY the corrected final answer text. No markdown.\n\n"
            f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft_answer}\n\nTASKS JSON CONTEXT:\n{ctx_text}\n"
        )
        messages = [
            {"role": "system", "content": "Return ONLY final answer text. No markdown."},
            {"role": "user", "content": user_payload},
        ]
        data = _call_openrouter(model, messages, temperature=0.0)
        if not data:
            return draft_answer
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return draft_answer


def generate_refusal(user_name: str) -> str:
    """Generate a spooky and funny refusal message for unauthorized users."""
    prompt = (
        f"You are a spooky, slightly unhinged AI guardian. A stranger named '{user_name}' tried to access your master's bot. "
        "Generate a short, scary, yet funny message telling them to leave immediately. "
        "Mention that this place is not for them. Use emojis. Be creative. Answer in Russian."
    )
    messages = [
        {"role": "system", "content": "You are a spooky AI. Answer in Russian."},
        {"role": "user", "content": prompt},
    ]
    for m in _get_models_list():
        try:
            data = _call_openrouter(m, messages, temperature=0.8)
            if not data:
                continue
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return "Уходи... Тебе здесь не рады... 👻"


def personalize_message(base_message: str, context: str = "") -> str:
    """Generate a personalized, creative message based on base message and context.
    Uses AGENT_ROLE and USER_PROFILE to customize the response."""
    role = os.getenv("AGENT_ROLE", "").strip()
    user_name = os.getenv("USER_NAME", "").strip()
    
    # If no role defined, return base message
    if not role:
        return base_message
    
    prompt = (
        f"You are {role}. "
        f"Rewrite the following message in a creative, personalized way that matches your personality. "
        f"Keep it short (1-2 sentences max). Use emojis sparingly. Answer in Russian.\n\n"
        f"Original message: {base_message}\n"
    )
    if context:
        prompt += f"Context: {context}\n"
    if user_name:
        prompt += f"User name: {user_name}\n"
    
    messages = [
        {"role": "system", "content": "Answer in Russian. Be creative but concise."},
        {"role": "user", "content": prompt},
    ]
    
    for m in _get_models_list():
        try:
            data = _call_openrouter(m, messages, temperature=0.7)
            if not data:
                continue
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    
    # Fallback to base message
    return base_message
