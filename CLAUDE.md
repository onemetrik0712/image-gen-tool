# Project: OneMetrik Image Generator (omgen)

## Purpose
CLI tool to generate ad creative variations using Replicate API (Flux model).
Built as a learning project for API integration with Claude Code.

## Tech Stack
- Python 3.11+
- Replicate API (Flux Schnell model)
- python-dotenv for environment variables

## Commands
- Run: `python src/generate.py "your prompt here" --variations 5`
- Test: `pytest tests/`
- Install deps: `pip install -r requirements.txt`

## Project Structure
```
image-gen-tool/
├── CLAUDE.md
├── .env.example
├── .env (never commit)
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   └── generate.py
├── output/
└── tests/
    └── test_generate.py
```

## Environment Variables
- REPLICATE_API_TOKEN: API key from replicate.com

## Coding Standards
- Use type hints on functions
- Add docstrings explaining what functions do
- Handle API errors gracefully with clear messages
- Print progress feedback for user

## Workflow Rules
1. Plan before coding
2. Test each feature before moving on
3. Keep functions focused and under 30 lines
4. User-friendly error messages (not raw tracebacks)

## Current Phase
Phase 1 MVP - Basic generation with prompt enhancement option

## Do Not
- Hardcode API keys
- Commit .env file
- Skip error handling for API calls
- Generate more than 10 images in one run (cost control)
