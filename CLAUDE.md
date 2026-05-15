# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

- **Weather module**: Switched to Open-Meteo primary (free, no key, true daily high/low), OWM fallback
- **All tests passing**: 95 unit tests across `tests/test_agent_logic.py`, `tests/test_location.py`, `tests/test_database.py`
- **Vacation mode**: Mode-aware targets — 85F (cool), 60F (heat); guardrails widen to 60-85F
- **Do NOT push**: `testHarness/` folder (added to `.gitignore`)

## gstack

- For all web browsing, always use the `/browse` skill from gstack. Never use `mcp__claude-in-chrome__*` tools.

### Available gstack skills

- `/office-hours` - Office hours
- `/plan-ceo-review` - Plan CEO review
- `/plan-eng-review` - Plan engineering review
- `/plan-design-review` - Plan design review
- `/design-consultation` - Design consultation
- `/review` - Code review
- `/ship` - Ship
- `/land-and-deploy` - Land and deploy
- `/canary` - Canary
- `/benchmark` - Benchmark
- `/browse` - Web browsing
- `/qa` - QA
- `/qa-only` - QA only
- `/design-review` - Design review
- `/setup-browser-cookies` - Setup browser cookies
- `/setup-deploy` - Setup deploy
- `/retro` - Retro
- `/investigate` - Investigate
- `/document-release` - Document release
- `/codex` - Codex
- `/cso` - CSO
- `/autoplan` - Auto plan
- `/careful` - Careful mode
- `/freeze` - Freeze
- `/guard` - Guard
- `/unfreeze` - Unfreeze
- `/gstack-upgrade` - Upgrade gstack
