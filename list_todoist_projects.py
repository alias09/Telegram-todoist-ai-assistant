from __future__ import annotations

import json
import sys

from dotenv import load_dotenv
from todoist_client import list_projects, TodoistError


def main() -> int:
    # Load .env from project root
    load_dotenv()
    try:
        projects = list_projects()
    except TodoistError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    # Print id and name for easy mapping into PROJECTS env
    for p in projects:
        pid = p.get("id")
        name = p.get("name") or p.get("project") or ""
        print(f"{name}: {pid}")
    # Also dump full JSON if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(projects, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
