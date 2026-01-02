# LLM Chatter

A simple GUI application for having two LLM models chat with each other via Ollama.

## Requirements

- Python 3.7+
- Ollama running locally (customizable, default: http://localhost:11434)

## Installation

(create/activate your venv if necessary)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running and you have at least one model installed.

## Usage

Run the application:
```bash
python llm_chatter.py
```

1. Select two models from the dropdown menus
2. Optionally customize system prompts for each model
3. Set the initial message to start the conversation
4. Adjust temperature and max history length as needed
5. Click "Start Conversation" to begin

## Features

- Two models chat back and forth automatically
- Dark mode UI
- Markdown rendering in chat messages
- Configurable temperature for response variation
- Conversation history management to prevent repetitive loops
- Adjustable delay between messages

## Current Problems

LLMs currently have a tendency to repeat themselves, especially models with small context windows.