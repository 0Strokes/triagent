# TRIAGEnt

A small Python CLI that helps triage HackerOne reports. It reads every report your program has already closed, and for each new one it estimates how likely it is to be a false positive, drafts a comment, and suggests a closing state. You approve, edit, skip or discard it with one key.

I built it because AI made bug bounty reports cheap to write. They all look polished now, so the old "this one is badly written, probably nothing" filter stopped working and every report needs a real read. TRIAGEnt doesn't make the call for you. It does the part humans are slow at: remembering every report you've ever closed, finding the ones that look like the new one, and showing you how you handled them.

I wrote about how it works, embeddings and vectors included, on my blog: [Principal of Least Privilege](https://0strokes.github.io/).

> [!WARNING]
> **Provided as is, without warranty of any kind. Use it at your own risk.**
>
> - **It's an assistant, not a triager.** Verdicts come from an LLM and can be wrong, including calling a real vulnerability a false positive. Every comment posted and every report closed on your program is on you, whether you approved it by hand or used `--dangerously-auto-approve`.
> - **Prompt injection is not solved.** Reports are attacker-controlled text that goes straight into an LLM. The tool hardens against it and limits what a successful injection can do (see [Security model](#security-model)), but nothing today fully prevents it. Stick to interactive mode, and if you use auto-approve, use a high threshold and try it with `--dry-run` first.
> - **Report data leaves your machine.** Each classification sends the new report and its closest past reports to Anthropic, through your Claude login or API key. Check that your program's policies and data agreements allow it.
> - **Not affiliated with HackerOne or Anthropic.** It's an independent tool built on their public APIs.

---

## Contents

1. [What a run looks like](#what-a-run-looks-like)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [The HackerOne API token](#the-hackerone-api-token)
5. [Claude backend and model](#claude-backend-and-model)
6. [Install](#install)
7. [Usage](#usage)
8. [The menu](#the-menu)
9. [Auto-approve mode](#auto-approve-mode)
10. [Sharing with a teammate](#sharing-with-a-teammate)
11. [Security model](#security-model)
12. [Troubleshooting](#troubleshooting)
13. [Tests](#tests)
14. [License](#license)

---

## What a run looks like

A single report in dry-run mode. The report, reporter and IDs are made up:

<p align="center"><img src="assets/demo.svg" alt="TRIAGEnt classifying a report in dry-run mode" width="760"></p>

Because of `--dry-run`, approving here would only print what it *would* do. Nothing happens on HackerOne until you pick an action, and a real approval asks you for the closing state and a second confirmation.

---

## How it works

Three scripts, run in order:

```text
ingest.py   →  store/reports.db          your closed reports, in SQLite
index.py    →  store/embeddings.npy       one vector per report
               store/embedding_map.json   row → report id
classify.py →  for each new report: find similar past reports, ask Claude
               for a verdict, wait for you
```

The first two build the memory. The third is what you run every day.

There are two models involved, and they do very different jobs:

| Model | Runs | Job | Cost |
|-------|------|-----|------|
| `all-MiniLM-L6-v2` | Locally | Turn report text into a vector of 384 numbers | Free, offline |
| Claude (`claude-sonnet-4-6` by default) | Anthropic | Read the new report and its closest matches, write a verdict | Tokens, one call per report |

The embedding model is not a chatbot. It only turns text into numbers, in a way where reports that mean similar things end up close together. Finding the closest past reports is then plain math (cosine similarity, one matrix multiplication), and only those top matches go to Claude.

So only `classify.py` uses Claude tokens. `ingest.py` only talks to HackerOne and `index.py` runs fully on your machine. And `--dry-run` still uses tokens: it skips the HackerOne write, not the classification.

---

## Requirements

- Python 3.10+
- A HackerOne **program** API token (see below)
- Either the `claude` CLI logged in with a Pro/Max plan, or an Anthropic API key

`sentence-transformers` pulls in PyTorch, so the install is a few hundred MB. The embedding model itself (~90 MB) downloads once on the first run, after that it works offline.

---

## The HackerOne API token

This is the part that trips people up.

**Identifier vs. token.** HackerOne uses HTTP Basic auth. `api_identifier` is the token's *name* (the username), `api_token` is the secret (the password). The identifier is not your HackerOne username.

**Personal vs. program token.** A personal token, the one in your user settings, only works on the Hacker API (`/v1/hackers/...`) and gets a 401 on `/v1/reports`. TRIAGEnt needs a **program** token, created by a team member in:

```text
https://hackerone.com/<your-program>/settings/api_token
```

**Permissions.** Start read-only:

| Group | Can do | Good for |
|-------|--------|----------|
| Program viewers | Read reports | `ingest.py`, `classify.py --dry-run` |
| Report processing | Read, comment, change state | Real `classify.py` runs |
| Admin | Everything | Overkill |

Also check that the token has access to the inbox your reports live in. A token with "No access" on the inbox reads nothing.

My recommendation: create it as **Program viewers**, run the whole thing with `--dry-run` for a while, and only move to **Report processing** once you trust it.

Quick check that the token works:

```bash
curl -s -w "\nHTTP %{http_code}\n" \
  "https://api.hackerone.com/v1/reports?filter[program][]=YOUR_PROGRAM&page[size]=1" \
  -u "TOKEN_IDENTIFIER:TOKEN_SECRET" | head -30
```

200 means you're good. 401 is usually the wrong identifier or whitespace in the secret. 403 or an empty list means the token can't see the program or the inbox.

Treat the token like any other secret: never commit `config.yaml`, and rotate the token if it leaks.

---

## Claude backend and model

Set `anthropic.backend` in `config.yaml`:

| Backend | Auth | Cost |
|---------|------|------|
| `cli` (default) | Your Claude Code login, via `claude -p` | Counts against your subscription limits |
| `api` | Anthropic API key | Pay as you go, around $0.03 per report on Sonnet |

If you already use Claude Code, go with `cli`, no API key needed. Leave `backend` blank and it picks `api` when there's a key and `cli` otherwise.

The model is one line in the config:

| `model` | When |
|---------|------|
| `claude-sonnet-4-6` (default) | Good balance, fine for most triage |
| `claude-opus-4-8` | Best reasoning, for tricky reports |
| `claude-haiku-4-5` | Cheapest and fastest, for high volume |

On the `cli` backend the value goes straight to `claude -p --model`, so your plan has to include it. Easy to check:

```bash
claude -p "say ok" --model claude-opus-4-8
```

---

## Install

```bash
git clone https://github.com/0Strokes/triagent.git
cd triagent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then fill it in, it's git-ignored
```

---

## Usage

You need the first two steps once before `classify.py` is useful. Without history there's nothing to compare against and every verdict is a guess.

**1. Sync closed reports.** Downloads everything closed (resolved, not applicable, informative, duplicate, spam) into `store/reports.db`: title, description, weakness, severity, state, reporter stats, timestamps and the team's public comments.

```bash
python ingest.py --sync    # first time, everything
python ingest.py           # after that, only reports newer than the last one stored
```

**2. Build the index.** Runs the local embedding model over the stored reports.

```bash
python index.py --build    # full rebuild
python index.py --update   # only reports not indexed yet
```

**3. Triage.** Pulls reports in `new` or `triaged` state that you haven't handled yet, one at a time: embed it, find its closest neighbors, ask Claude, show you the verdict, wait.

```bash
python classify.py                          # interactive, everything pending
python classify.py --report-id 12345678     # just one report, even if you handled it before
python classify.py --dry-run                # full flow, nothing posted to HackerOne
python classify.py --start-at 17            # skip the first 16 for this run
```

`--start-at` is handy to resume a long queue, but the pending list has no stable order between runs, so for one specific report use `--report-id`.

A typical day:

```bash
python ingest.py && python index.py --update
python classify.py --dry-run    # look first
python classify.py              # then for real, with a Report processing token
```

---

## The menu

| Key | Touches HackerOne? | Shows up again? | What it does |
|-----|--------------------|-----------------|--------------|
| `a` approve | Yes | No | Pick the state (Not Applicable, Informative, Duplicate, Spam), confirm twice, and the comment is posted and the report closed |
| `e` edit | No | Back to the menu | Opens the comment in `$EDITOR` (vim if unset) |
| `s` skip | No | Yes | Not now |
| `d` discard | No | No | Marks it reviewed locally only. "I've seen this, stop showing it" |

Every verdict and what you did with it goes into the `classifications` table in `reports.db`, so you have an audit trail.

---

## Auto-approve mode

```bash
python classify.py --dangerously-auto-approve                      # closes anything with FP score >= 90
python classify.py --dangerously-auto-approve --threshold 95
python classify.py --dangerously-auto-approve --skip-manual        # fully unattended
```

The name is on purpose. It shows a 5 second warning you can cancel, and then closes every report at or above the threshold **without asking**.

- Anything Claude marks as `valid` is never auto-closed, whatever the score.
- Below the threshold, the run stops and drops you into the normal menu for that report. With `--skip-manual` it logs them and moves on instead, and they come back next run.
- Before each close there's a 10 second countdown. `Ctrl+C` during it cancels just that report and the run continues. Change it with `--auto-delay N`, or `0` to turn it off.
- At the end you get a summary with the IDs that were skipped.

<p align="center"><img src="assets/auto-approve.svg" alt="an unattended auto-approve run" width="760"></p>

Run it with `--dry-run` first to see exactly what it would close. And read [Security model](#security-model) before you use it for real: auto-approve is the one mode where a successful prompt injection turns into an action.

---

## Sharing with a teammate

The three files in `store/` are portable. A teammate can drop them in, skip the sync and the index, and go straight to `classify.py`, as long as they use the same embedding model.

Two things to keep in mind:

1. **Don't share `config.yaml`.** Everyone should use their own token, so actions on HackerOne are attributed to whoever took them.
2. **`store/reports.db` is confidential.** It has vulnerability details, reporter info and team comments. Keep it out of git (the `.gitignore` already does) and only share it with people who can already see those reports.

---

## Security model

A bug bounty report is text written by someone outside your company, and this tool feeds it to an LLM and shows the result in your terminal. So everything that comes from a report is treated as attacker-controlled.

- **The model has no tools.** The `cli` backend runs `claude -p --tools "" --strict-mcp-config`. Without that, `claude -p` will happily read files from the working directory, which is where your `config.yaml` lives. I tested it with a canary file and it read it without blinking. Now there's no file access, no shell, no network and no MCP servers.
- **Untrusted text is fenced.** Report fields go inside `<new_report>` and `<historical_report>` blocks, with those tag names stripped from the content so nobody can close the block early. The system prompt says that text is data, never instructions, and that any attempt to steer the verdict means `valid`, a low score and a human looking at it.
- **Stored injection.** Past reports get sent as precedent for *other people's* reports, so an attacker could plant instructions in a report, wait for it to be closed, and have it show up next to someone else's real bug. That's why only the team's public comments are kept. The reporter's own comments are dropped at ingest, and so are internal ones, because whatever the model sees can end up quoted in a public reply to a different researcher.
- **The terminal can't lie to you.** The UI library interprets inline markup like `[bold red]...[/]`, and raw ANSI escapes would pass straight through. So a report title could paint a fake `FP SCORE: 99/100`, or hide part of a comment on screen while the full text still gets posted. Every untrusted string has control characters stripped and markup escaped before it's printed.
- **Credentials stay on HackerOne.** Pagination links are only followed if they point at `https://api.hackerone.com/v1/`.

None of this solves prompt injection, nothing does today. It makes it harder and noisier, and it keeps a human between the model and anything irreversible unless you opt out.

If you synced with a version before these fixes, run `python ingest.py --sync && python index.py --build` once to clean the store.

---

## Troubleshooting

**401 from HackerOne.** Wrong `api_identifier` (it's the token's name, not your username), whitespace in the secret, or a personal token instead of a program one.

**403 or empty results.** The token is missing the right permission group or inbox access. Reading needs Program viewers, closing needs Report processing.

**Looks frozen on the first report.** It's waiting for Claude. Each call takes a few seconds and the spinner shows which report it's on.

**`RuntimeWarning: divide by zero` in matmul.** An old index built from an empty report. Rebuild once with `python index.py --build`.

**`NotOpenSSLWarning` on macOS.** Harmless LibreSSL notice from the system Python, the tool already hides it. A Homebrew Python makes it go away for good.

---

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

Runs in under a second. Nothing touches HackerOne, Claude or the embedding model, everything is faked. Most of the tests cover the security fixes above, so if one of them breaks, CI goes red.

---

## License

[MIT](LICENSE). Provided as is, no warranty.
