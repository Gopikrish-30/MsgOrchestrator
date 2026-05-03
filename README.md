# Vera Bot

Vera Bot is a FastAPI service that turns merchant, category, trigger, and customer context into short WhatsApp-ready messages. The current version is tuned for the Magicpin AI Challenge: it is deterministic, context-grounded, judge-friendly, and ready for local execution or deployment.

## What This Build Includes

- Proactive message generation from live merchant, category, trigger, and customer context
- Versioned, idempotent context storage with stale-version rejection
- Multi-turn reply handling with auto-reply detection, action intent detection, offtopic redirects, and graceful exits
- Judge-style action probes are handled even when trigger context has not been populated yet
- Category-specific voice rules so messages feel native to each vertical
- Trigger-specific dispatch logic with richer facts and dynamic instruction generation
- Expanded dataset support and canonical `submission.jsonl` generation
- Judge cleanup support through `/v1/teardown`

## Current API Surface

- `GET /v1/healthz` - service status, uptime, and loaded context counts
- `GET /v1/metadata` - team info, model name, version, approach summary, and rubric targets
- `POST /v1/context` - store category, merchant, customer, and trigger context with version control
- `POST /v1/tick` - compose proactive WhatsApp actions from available trigger IDs
- `POST /v1/reply` - process merchant replies and return `send`, `wait`, or `end`
- `POST /v1/teardown` - clear in-memory state for judge resets and test cleanup

## What Changed Recently

This repo has been updated beyond the original baseline in a few important ways:

- Model upgraded from the older 8B default to `meta-llama/llama-4-scout-17b-16e-instruct`
- Prompting improved so the detailed trigger-specific instruction is actually used
- Response budget expanded to reduce JSON truncation
- Facts passed to the model are trimmed to keep the prompt focused
- Reply composition now returns an explicit `action` field
- `/v1/teardown` now exists and clears conversation and context state
- Conversation handling now supports graceful exit after repeated auto-replies
- Action probes are answered even without trigger context, preventing empty replies in judge simulations
- Reply handling now includes better behavior for action, question, and offtopic intents
- Fallback responses are more specific and context-aware

## Core Behavior

### Composition flow

1. The app loads seed context at startup.
2. `POST /v1/context` stores versioned category, merchant, trigger, and customer payloads.
3. `POST /v1/tick` resolves the requested contexts, builds trigger-aware instructions, and calls Groq for composition.
4. The model returns a JSON response with body, CTA, rationale, and send mode.
5. The response is converted into an action payload for the judge or downstream harness.

### Reply flow

1. `POST /v1/reply` records the incoming merchant message.
2. The conversation state machine checks for auto-replies, action intent, exit intent, offtopic content, and unanswered-message thresholds.
3. If a judge-style action probe arrives before `/v1/tick`, the service still returns a next-step response instead of hard-failing on missing trigger context.
4. The service either sends a follow-up, ends gracefully, or provides a contextual response.
5. The reply path now supports `action: send | wait | end` instead of always falling back to send.

## Key Implementation Details

### `main_enhanced.py`

- FastAPI entrypoint used by `uvicorn`, Docker, and the PaaS process file
- Hosts all API routes and request/response models
- Tracks conversation metadata for `/v1/reply` lookups
- Includes teardown cleanup for judge test cycles
- Implements question and offtopic handlers so replies do not collapse into a generic default

### `composer_enhanced.py`

- Uses the Scout model configured through `GROQ_MODEL`
- Builds trigger-specific instructions dynamically instead of relying only on static templates
- Trims large facts before JSON serialization to preserve output budget
- Raises the token cap to reduce truncated responses
- Produces a richer fallback response when the model cannot be used

### `conversation_enhanced.py`

- Maintains per-conversation turn history and counters
- Detects auto-replies with pattern matching and repeat detection
- Separates action, exit, question, and offtopic intent
- Prevents a “thanks” false positive from overriding real action intent
- Supports graceful exit messages and alternate follow-up messages

### `context_store.py`

- Thread-safe in-memory storage
- Idempotent same-version updates
- Rejects stale versions
- Exposes `clear_all()` for teardown and keeps `clear()` as a compatibility alias

## Dataset And Submission

The project ships with an expanded dataset for challenge evaluation:

- 50 merchants
- 200 customers
- 100 triggers
- Canonical submission generation through `generate_submission.py`
- `submission.jsonl` contains the 30 required rows for the challenge output format

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

4. Add your Groq API key and project metadata to `.env`.

5. Start the app.

```bash
uvicorn main_enhanced:app --host 0.0.0.0 --port 8080
```

## Environment Variables

- `GROQ_API_KEY` - required for model access
- `GROQ_MODEL` - defaults to `meta-llama/llama-4-scout-17b-16e-instruct`
- `GROQ_BASE_URL` - Groq API base URL
- `TEAM_NAME` - displayed in `/v1/metadata`
- `TEAM_MEMBERS` - displayed in `/v1/metadata`
- `VERSION` - optional build/version tag

## Local Testing

Run the full validation flow after setup:

```bash
python test_suite.py
python dataset/generate_dataset.py --seed-dir dataset --out expanded
python generate_submission.py
```

For a quick runtime check, verify these endpoints after startup:

- `/v1/healthz`
- `/v1/metadata`
- `/v1/context`
- `/v1/tick`
- `/v1/reply`
- `/v1/teardown`

## Deployment

The repository is configured to run with `main_enhanced:app`.

### Docker

```bash
docker build -t vera-bot .
docker run -p 8080:8080 --env-file .env vera-bot
```

### PaaS

Set the same environment variables in your hosting platform and use the `Procfile` or Dockerfile as your entrypoint.

## Deploy to Railway (recommended)

This is the easiest, repeatable method: connect your GitHub repo to Railway and enable automatic deploys from `master`.

1. Prepare and push your changes to GitHub (from `vera-bot/`):

```bash
# from inside vera-bot/
git status
git add .
git commit -m "chore: final README + deployment instructions"
git push origin master
```

2. Create a Railway project (web UI is simplest):

- Open https://railway.app and sign in with GitHub.
- Click **New Project → Deploy from GitHub**.
- Select the `Gopikrish-30/MsgOrchestrator` repository and the `master` branch.

3. Configure Railway environment variables (in Railway project → Settings → Variables):

```
GROQ_API_KEY=sk-...                    # required
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_BASE_URL=https://api.groq.com/openai/v1
TEAM_NAME=Vera Engine
TEAM_MEMBERS=Your Name
CONTACT_EMAIL=you@example.com
PORT=8080
```

4. Set the start command and build settings (Railway may auto-detect):

- **Start Command:** `uvicorn main_enhanced:app --host 0.0.0.0 --port 8080`
- **Build Command:** `pip install -r requirements.txt`

5. Deploy

- Trigger a deploy from the Railway UI or push a new commit to `master`.
- Monitor build logs in Railway; once healthy, your service will be public at the Railway-assigned URL.

Railway CLI alternative (optional):

```bash
# Install CLI: https://docs.railway.app/develop/cli
railway login
railway init         # follow prompts to link/create a project
railway up           # deploy current repo
```

Why this method: GitHub integration + Railway gives automatic CI, secrets management, and a simple UI to see logs and redeploy — minimal ops overhead and ideal for fast iteration.


## Notes On Output Quality

- The model prompt is structured to favor specificity, category fit, merchant fit, trigger relevance, and engagement
- Trigger-specific facts are prioritized so the bot can mention concrete numbers, offers, and context
- The system is designed to avoid generic fallback text unless generation fails

## Evaluation

See [testcases.md](testcases.md) for the validation summary and end-to-end test results.
