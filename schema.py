from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ReminderItem(BaseModel):
    when: Optional[Literal["relative", "absolute"]] = None
    offset: Optional[str] = None  # ISO 8601 duration, e.g., PT30M
    at: Optional[str] = None  # ISO 8601 datetime in UTC
    note: Optional[str] = None


class NewTask(BaseModel):
    # Обязательные поля по ТЗ: title, body, created_at, project, labels, priority, deadline, direction
    title: str
    body: Optional[str] = None
    created_at: Optional[str] = Field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat()+"Z")
    project: Optional[str] = None  # Предопределённый список, но разрешаем произвольную строку
    labels: List[str] = Field(default_factory=list)
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    deadline: Optional[str] = None  # ISO 8601 UTC
    direction: Optional[Literal["from_me", "to_me"]] = None  # От меня / ко мне
    source_text: Optional[str] = None


class TaskChanges(BaseModel):
    status: Optional[Literal["todo", "in_progress", "blocked", "done"]] = None
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    labels_add: List[str] = Field(default_factory=list)
    labels_remove: List[str] = Field(default_factory=list)
    assignee: Optional[str] = None
    deadline: Optional[str] = None


class TaskUpdate(BaseModel):
    target: str
    changes: TaskChanges
    source_text: Optional[str] = None


class Reminder(BaseModel):
    title: str
    at: Optional[str] = None
    offset: Optional[str] = None
    repeat: Optional[Literal["none", "daily", "weekly", "monthly"]] = None
    source_text: Optional[str] = None


class Meta(BaseModel):
    language: Optional[str] = None
    parsed_at: Optional[str] = None
    confidence: Optional[float] = None
    intents: Optional[List[Dict[str, Any]]] = None


class ExtractionResult(BaseModel):
    tasks_new: List[NewTask] = Field(default_factory=list)
    tasks_updates: List[TaskUpdate] = Field(default_factory=list)
    reminders: List[Reminder] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    meta: Meta = Field(default_factory=lambda: Meta(parsed_at=datetime.utcnow().replace(microsecond=0).isoformat()+"Z"))


EMPTY_RESULT = ExtractionResult()
