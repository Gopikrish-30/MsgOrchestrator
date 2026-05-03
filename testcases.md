# Vera Bot — Test Cases and Evaluation

## Summary

The Vera Bot implementation was validated across the full runtime path:

- API endpoints: health, metadata, context, tick, reply
- Core modules: context store, composer, conversation state machine, seed loader
- Artifact generation: `submission.jsonl`
- Fresh-conversation auto-reply handling
- Judge-style action probes without preloaded trigger context

## Results

- `GET /v1/healthz`: passed
- `GET /v1/metadata`: passed
- `POST /v1/context`: passed, including same-version idempotence and stale-version rejection
- `POST /v1/tick`: passed, returns actions from available triggers
- `POST /v1/reply`: passed, including brand-new auto-reply handling returning `wait`
- `POST /v1/reply`: passed, including action probes that arrive before `/v1/tick`
- `POST /v1/teardown`: passed, clears in-memory state for judge resets
- `generate_submission.py`: passed, writes exactly 30 entries to `submission.jsonl`
- Python syntax checks: passed for all remaining source files

## Key Behaviors Verified

- Fresh auto-reply messages are detected from content alone and return `wait` before escalating to `end`
- Action intent is preserved even when trigger metadata has not been populated yet
- Metadata reports the configured Groq model dynamically
- Submission generation uses the expanded dataset and canonical 30 test pairs
- The app starts from `main_enhanced.py`

## Notes

- The current runtime model is `meta-llama/llama-4-scout-17b-16e-instruct`
- The repository now keeps only the main README and this evaluation file as markdown documentation
