<!-- Guidance for AI coding assistants: concise, repo-specific rules -->

# Copilot / AI assistant instructions for this repository

This repository is a small, personal Python learning workspace (MLOps basics). The goal of these instructions is to give an AI coding agent the exact, discoverable context needed to make safe, useful edits.

Key facts
- This is not a packaged application or library. It contains short, standalone Python examples under `python_study/`.
- No dependency manifests (requirements.txt/pyproject.toml) or CI are present. Avoid adding them unless explicitly asked.

Layout to reference
- `python_study/numpy/how to use` — notes / examples for NumPy usage.
- `python_study/others/test1` — a small, runnable demo (datetime usage and basic printing). Example lines worth referencing:
  - `b = datetime.now()`
  - `c = datetime.now().timestamp()`
  - commented example: `# print(datetime.strptime(c,'%Y-%m-%d %H:%M:%S'))`

What the agent should do
- Prefer minimal, self-contained changes that keep examples runnable with the system Python (standard library only).
- When adding a new example file, place it in `python_study/<topic>/` and add a one-line comment at the top describing purpose and how to run it.
- When changing an existing example, preserve its didactic intent: keep prints/simple outputs and avoid infrastructure additions (logging frameworks, CI, packaging) unless requested.

Style and patterns observed
- Scripts are direct, top-level scripts (not modules). Use `if __name__ == '__main__':` only when adding a slightly larger example that benefits from importability.
- Use standard-library imports and explicit names (e.g., `from datetime import datetime`) as in `test1`.
- Demonstrations show simple print-based verification. If you add assertions or test harnesses, leave them optional and documented in-file.

Tests & verification
- There are no automated tests. After edits, run the script manually (example):

```powershell
# from project root
python python_study/others/test1
```

What not to change or add without approval
- Do not convert the repo into a package (no setup.py, pyproject.toml) or add dependency/CI files.
- Do not replace simple prints with heavy frameworks or change the learning intent of examples.

If you need more structure (tests, CI, packaging), ask before implementing — provide a short rationale and a minimal plan.

If any of these rules are unclear, point to the file or example you want changed and I'll update these instructions.
