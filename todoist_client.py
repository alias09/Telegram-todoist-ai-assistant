from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional
import logging

import httpx

API_BASE = "https://api.todoist.com/rest/v2"

logger = logging.getLogger(__name__)


class TodoistError(Exception):
    pass


def _get_token() -> str:
    token = os.getenv("TODOIST_API_TOKEN", "").strip()
    if not token:
        raise TodoistError("TODOIST_API_TOKEN is not set")
    return token


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
        "X-Request-Id": str(uuid.uuid4()),
    }


def list_projects() -> List[Dict[str, Any]]:
    url = f"{API_BASE}/projects"
    headers = _headers()
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug("Todoist.list_projects request url=%s headers=%s", url, safe_headers)
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=headers)
        if r.status_code >= 400:
            logger.error("Todoist.list_projects failed status=%s body=%s", r.status_code, r.text)
        r.raise_for_status()
        logger.debug("Todoist.list_projects success count=%s", len(r.json() or []))
        return r.json()


def create_task(
    *,
    content: str,
    description: Optional[str] = None,
    project_id: Optional[str] = None,
    labels: Optional[List[str]] = None,
    priority: Optional[int] = None,
    due_datetime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create task via Todoist API v1.
    - content: title
    - description: additional text
    - project_id: project id (str)
    - labels: list of label names
    - priority: 1..4
    - due_datetime: RFC3339 datetime (UTC with 'Z' accepted)
    """
    url = f"{API_BASE}/tasks"
    payload: Dict[str, Any] = {
        "content": content,
    }
    if description:
        payload["description"] = description
    if project_id:
        payload["project_id"] = project_id
    if labels:
        payload["labels"] = labels
    if priority:
        payload["priority"] = int(priority)
    if due_datetime:
        payload["due_datetime"] = due_datetime
    headers = _headers()
    # Sanitize headers for logging
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug(
        "Todoist.create_task request url=%s payload=%s headers=%s",
        url,
        payload,
        safe_headers,
    )
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            # Log body for diagnostics, then raise
            try:
                body = r.text
            except Exception:
                body = "<no body>"
            logger.error(
                "Todoist.create_task failed status=%s body=%s payload=%s",
                r.status_code,
                body,
                payload,
            )
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            data = {}
        logger.debug(
            "Todoist.create_task success task_id=%s project_id=%s",
            data.get("id"),
            data.get("project_id"),
        )
        return data


def get_tasks(*, filter: Optional[str] = None, project_id: Optional[str] = None, label: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get active tasks (REST v2) with optional server-side filters.
    Supported query params: filter, project_id, label
    """
    url = f"{API_BASE}/tasks"
    headers = _headers()
    params: Dict[str, str] = {}
    if filter:
        params["filter"] = filter
    if project_id:
        params["project_id"] = project_id
    if label:
        params["label"] = label
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug("Todoist.get_tasks request url=%s params=%s headers=%s", url, params, safe_headers)
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=headers, params=params or None)
        if r.status_code >= 400:
            logger.error("Todoist.get_tasks failed status=%s body=%s", r.status_code, r.text)
        r.raise_for_status()
        data = r.json()
        logger.debug("Todoist.get_tasks success count=%s", len(data or []))
        return data


def update_task(
    task_id: str,
    *,
    content: Optional[str] = None,
    description: Optional[str] = None,
    labels: Optional[List[str]] = None,
    priority: Optional[int] = None,
    due_datetime: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Update task fields (REST v2 POST /tasks/{id})."""
    url = f"{API_BASE}/tasks/{task_id}"
    payload: Dict[str, Any] = {}
    if content is not None:
        payload["content"] = content
    if description is not None:
        payload["description"] = description
    if labels is not None:
        payload["labels"] = labels
    if priority is not None:
        payload["priority"] = int(priority)
    if due_datetime is not None:
        payload["due_datetime"] = due_datetime
    if project_id is not None:
        payload["project_id"] = project_id
    headers = _headers()
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug("Todoist.update_task request url=%s payload=%s headers=%s", url, payload, safe_headers)
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.error("Todoist.update_task failed status=%s body=%s payload=%s", r.status_code, r.text, payload)
        r.raise_for_status()
        data = r.json() if r.text else {}
        logger.debug("Todoist.update_task success body=%s", data)
        return data


def move_task(task_id: str, *, project_id: str) -> Dict[str, Any]:
    """Move task to another project via update (REST v2 POST /tasks/{id})."""
    return update_task(task_id, project_id=project_id)


def close_task(task_id: str) -> None:
    url = f"{API_BASE}/tasks/{task_id}/close"
    headers = _headers()
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug("Todoist.close_task request url=%s headers=%s", url, safe_headers)
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers)
        if r.status_code >= 400:
            logger.error("Todoist.close_task failed status=%s body=%s", r.status_code, r.text)
        r.raise_for_status()
        logger.debug("Todoist.close_task success")


def reopen_task(task_id: str) -> None:
    url = f"{API_BASE}/tasks/{task_id}/reopen"
    headers = _headers()
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug("Todoist.reopen_task request url=%s headers=%s", url, safe_headers)
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers)
        if r.status_code >= 400:
            logger.error("Todoist.reopen_task failed status=%s body=%s", r.status_code, r.text)
        r.raise_for_status()
        logger.debug("Todoist.reopen_task success")


def update_label(
    label_id: str,
    *,
    name: Optional[str] = None,
    color: Optional[str] = None,
    order: Optional[int] = None,
) -> Dict[str, Any]:
    """Update label entity (v1 POST /labels/{label_id})."""
    url = f"{API_BASE}/labels/{label_id}"
    payload: Dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if color is not None:
        payload["color"] = color
    if order is not None:
        payload["order"] = int(order)
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=_headers(), json=payload)
        r.raise_for_status()
        return r.json() if r.text else {}


def create_reminder(item_id: str, *, due: Dict[str, Any] | None = None, type: str = "custom") -> Dict[str, Any]:
    """Create a reminder for a task (REST v2 POST /reminders).
    due object example: {"string": "tomorrow at 10:00"} or {"datetime": "..."}
    """
    url = f"{API_BASE}/reminders"
    payload = {
        "item_id": item_id,
        "type": type,
    }
    if due:
        payload["due"] = due
    
    headers = _headers()
    safe_headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}
    logger.debug("Todoist.create_reminder request url=%s payload=%s headers=%s", url, payload, safe_headers)
    
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.error("Todoist.create_reminder failed status=%s body=%s", r.status_code, r.text)
        r.raise_for_status()
        return r.json()
