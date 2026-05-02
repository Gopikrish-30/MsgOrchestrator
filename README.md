# Vera Bot

Vera Bot is a FastAPI service that turns merchant, category, trigger, and customer context into short WhatsApp-ready messages. It is built to be deterministic, context-grounded, and easy to run locally or deploy.

## What It Does

- Composes proactive WhatsApp messages from live context
- Tracks versioned context updates with idempotent behavior
- Handles multi-turn replies with auto-reply and intent detection
- Uses category voice rules so messages feel relevant to each business vertical
- Generates the canonical 30 submission rows from the expanded dataset

## Core Features

- 5 endpoints: `/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`
- Groq-backed generation with `llama-3.1-8b-instant`
- Deterministic output by default (`temperature=0`)
- Thread-safe in-memory context storage
- Reply handling for `send`, `wait`, and `end`
- Explicit team metadata in `/v1/metadata`

## Project Structure

```text
vera-bot/
├── main_enhanced.py         FastAPI app and API routes
├── composer_enhanced.py     Message generation and reply composition
├── conversation_enhanced.py Conversation state machine
├── context_store.py         Versioned in-memory context store
├── seed_loader.py           Seed dataset loader
├── generate_submission.py   Builds submission.jsonl from the expanded dataset
├── test_suite.py            End-to-end validation script
├── testcases.md             Evaluation summary and test results
├── README.md                Project guide
├── requirements.txt         Python dependencies
├── Dockerfile               Container build file
├── Procfile                 Process entrypoint for PaaS deploys
├── .env.example             Environment template
└── dataset/                 Seed categories, merchants, customers, triggers
```

## Setup

1. Enter the app directory.

```bash
cd vera-bot
```

2. Create your local environment file.

```bash
cp .env.example .env
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Add your Groq API key and project details to `.env`.

5. Start the app.

```bash
uvicorn main_enhanced:app --host 0.0.0.0 --port 8080
```

## How To Add Team Details

Edit `.env` and set these fields:

```env
TEAM_NAME=Your Team Name
TEAM_MEMBERS=Member One,Member Two
CONTACT_EMAIL=your@email.com
GITHUB_REPO=https://github.com/your-org/your-repo
SUBMITTED_AT=2026-05-02T23:00:00Z
GROQ_MODEL=llama-3.1-8b-instant
```

These values are returned by `GET /v1/metadata`, so they are the main place to customize your submission identity.

## End-to-End Flow

1. `seed_loader.py` loads the seed dataset into memory on startup.
2. `POST /v1/context` stores category, merchant, customer, and trigger context.
3. `POST /v1/tick` selects triggers and composes proactive messages.
4. `POST /v1/reply` processes merchant replies and decides whether to send, wait, or end.
5. `generate_submission.py` expands the dataset and creates the canonical `submission.jsonl` artifact.

## API Endpoints

### `GET /v1/healthz`
Returns service status, uptime, and loaded context counts.

### `GET /v1/metadata`
Returns team name, team members, model name, version, contact email, and approach summary.

### `POST /v1/context`
Stores versioned context. Same version is idempotent; stale versions are rejected.

### `POST /v1/tick`
Composes proactive actions from the available trigger IDs and returns an `actions` array.

### `POST /v1/reply`
Handles merchant or customer replies and returns `action: send | wait | end`.

## Local Testing

Run the full validation flow after setup:

```bash
python test_suite.py
python dataset/generate_dataset.py --seed-dir dataset --out expanded
python generate_submission.py
```

## Deployment

The repo is configured to run with `main_enhanced:app`.

### Docker

```bash
docker build -t vera-bot .
docker run -p 8080:8080 --env-file .env vera-bot
```

### PaaS

Set the same environment variables in your hosting platform and use the `Procfile` or Dockerfile as your entrypoint.

## Configuration Notes

- Use `llama-3.1-8b-instant` unless you have a specific reason to change models.
- Keep `GROQ_API_KEY` in `.env`, not in source control.
- Do not commit `submission.jsonl`; it is generated from the dataset.

## Evaluation

See [testcases.md](testcases.md) for the validation summary and end-to-end test results.
