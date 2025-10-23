# AI Crypto Research Bot

Minimal scaffold for reproducible research pipelines, experiments registry, and walk-forward validation.

See configs/ for project settings.

## Quick Windows (PowerShell) setup

1. Create venv and install requirements (no Activate.ps1 required):

```powershell
cd C:\path\to\ai-crypto-research-bot
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Or use the helper scripts in `tooling\`:

```powershell
# one-time setup
tooling\run_setup.ps1
# run tests
tooling\run_tests.ps1
```

## Chat recovery & preventive tips

If you use an assistant extension (for example GitHub Copilot Chat) inside VS Code, conversations are often stored per VS Code window or per workspace. If you open a different folder or workspace the assistant may show a different (or empty) chat history.

Quick recovery steps:
- Reopen the original workspace file (for example `ai-crypto-research-bot.code-workspace`) from File → Open Recent.
- Copilot Chat saves sessions under your VS Code global storage. On Windows the files live at `%APPDATA%\\Code\\User\\globalStorage\\emptyWindowChatSessions` and `%APPDATA%\\Code\\User\\globalStorage\\github.copilot-chat` — you can copy the JSON files into your project to preserve them.
- If the extension supports cloud sync, sign into the same account and check the extension's History UI.

Prevent losing chats:
- Export or copy important conversations to a Markdown file inside the repo (useful for reproducibility). Example: copy the JSON to `copilot-session-YYYY.json` or paste a cleaned transcript into `docs/`.
- Keep a dedicated workspace file (`.code-workspace`) and open that instead of the raw folder to preserve workspace-scoped extension state.
- If your extension supports pinning or saving conversations, use it.

