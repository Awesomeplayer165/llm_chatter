# LLM Chatter

A simple GUI application for having two LLM models chat with each other. Supports **Ollama** and **OpenAI-compatible** endpoints (OpenAI, Azure, local proxies, etc.), selectable via a config file.

## Requirements

- Python 3.7+
- Either Ollama running locally, or an OpenAI-compatible API (with API key if required)

## Installation

(create/activate your venv if necessary)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Edit `config.json` to add or change providers (see **Configuration** below).

## Configuration

Provider endpoints are defined in **`config.json`** in the same folder as `llm_chatter.py`. You can define multiple providers and choose which one to use in the app.

Example `config.json`:

```json
{
  "default": "ollama-local",
  "providers": [
    {
      "id": "ollama-local",
      "name": "Ollama (local)",
      "type": "ollama",
      "base_url": "http://localhost:11434"
    },
    {
      "id": "openai-example",
      "name": "OpenAI / compatible",
      "type": "openai",
      "base_url": "https://api.openai.com/v1",
      "api_key": ""
    }
  ]
}
```

- **`default`** – ID of the provider selected when the app starts.
- **`providers`** – List of provider configs. Each has:
  - **`id`** – Unique key (e.g. `ollama-local`, `openai-example`).
  - **`name`** – Label shown in the app’s “Provider” dropdown.
  - **`type`** – `"ollama"` or `"openai"`.
  - **`base_url`** – API base URL (no trailing slash). For OpenAI: `https://api.openai.com/v1`; for Ollama: `http://localhost:11434`.
  - **`api_key`** – (OpenAI only) API key. Leave empty to use the `OPENAI_API_KEY` environment variable.

You can add more entries to `providers` (e.g. another Ollama host or a different OpenAI-compatible endpoint) and switch between them in the UI.

## Usage

Run the application:
```bash
python llm_chatter.py
```

1. Choose a **Provider** from the dropdown (from `config.json`).
2. Click **Refresh Models** to load models for that provider.
3. Select two models from the dropdown menus.
4. Optionally customize system prompts for each model.
5. Set the initial message and adjust temperature / max history / delay as needed.
6. Click **Start Conversation** to begin.

## Features

- **Multiple providers** – Ollama and OpenAI-compatible endpoints, switchable via `config.json` and the Provider dropdown.
- Two models chat back and forth automatically.
- Dark mode UI.
- Markdown rendering in chat messages.
- Configurable temperature for response variation.
- Conversation history management to prevent repetitive loops.
- Adjustable delay between messages.

## Current Problems

LLMs currently have a tendency to repeat themselves, especially models with small context windows.