# Product Decisions & Learnings

I built a local AI agent that manages my home's two Nest thermostats. My wife texts "it's cold" and the system reasons about indoor temp, outdoor forecast, time of day, and HVAC mode — then explains its reasoning in plain English before adjusting the temperature. All AI runs locally on my GPU. Zero cloud costs. It has been managing my home's climate autonomously since March 31, 2026.

This document captures the product thinking behind the project — the decisions, the tradeoffs, the pivots, and what I learned about shipping an AI product into a real environment with real users (me and my wife).

---

## Why This Exists

I own two basic (non-learning) Nest thermostats. I was spending ~5 minutes daily adjusting them manually. A Nest Learning Thermostat costs $250 per unit — $500 for both. I wanted the learning and optimization for free.

But more than saving money, I wanted to build something that actually *reasons*. Not a rule engine. Not a cron job with if-else. An agent that sees "it's 95F outside, the house is at 78F, and my wife just said she's heading home" and thinks: "I should pre-cool now so it's comfortable when she arrives."

**The success criterion I set on day one:** At least one friend says "whoa" when I show them the Telegram chat of the agent explaining its reasoning. That happened in the first week.

---

## Key Product Decisions

### 1. Local LLM over Cloud API

**Decision:** Run Gemma 3 1B (later Gemma 4 E2B) locally via llama.cpp. Zero cloud AI calls.

**Why — two reasons, both non-negotiable:**

- **Privacy.** My thermostat data includes when I'm home, when I'm sleeping, when I leave for dinner. Sending that to a cloud API creates an attack surface I'm not comfortable with. A compromised API key means someone knows my household patterns and can control my HVAC. Local means the data never leaves my machine.

- **Cost.** Cloud LLM APIs charge per token. At 3 evaluations per hour, 24/7, even a cheap model adds up. GPT-4o-mini at ~$0.15/M input tokens across ~2,200 calls/day would cost real money over months. Local inference costs electricity I'm already paying for. The marginal cost per decision is effectively zero.

This constraint shaped every downstream decision — model selection, prompt engineering, guardrail design. When your LLM is a 2B parameter model running on consumer hardware, you can't rely on it being smart. You have to make the *system* smart and let the LLM make the final call with heavy guidance.

### 2. SLM by Necessity, Not by Choice

**Decision:** Use a Small Language Model (Gemma 4 E2B — 2 billion parameters) instead of a 7B/13B/70B model.

**Why:** My computer is not a powerful machine. I don't have an A100 or even a 3090 — I have a consumer-grade GPU with limited VRAM. A 7B model is sluggish. A 13B model won't fit. A 2B model runs fast and leaves room for everything else I use my PC for. This wasn't a preference — it was a hardware constraint.

That constraint forced a design philosophy that turned out to be the project's biggest strength: **move all reasoning to Python, keep the prompt tiny.**

SLMs have small context windows (4096 tokens for my setup). Every token in the prompt matters. I can't send the model a 2,000-token essay about weather patterns, user history, comfort preferences, and scheduling rules, then ask it to reason from first principles. It would lose the plot halfway through.

Instead, the entire prompt fits in roughly 200-300 tokens:

```
You are a thermostat agent for the Upstairs Bedroom zone.

Indoor: 76F, 45% humidity
HVAC mode: cooling, target: 78F
Outdoor: 92F, Clear skies, high near 98F
Time: 02:30 PM Wednesday
Other zone Downstairs Kitchen: 74F, target=75F

Sleep time, summer. Indoor 76F is warm. Cool to 75F.

Respond with ONLY this JSON (action MUST be "set_temperature" or "no_change"):
{"action":"...","temperature":<65-80 or null>,"reasoning":"<brief>","message_to_user":"<optional or null>"}
```

That's it. The directive line — "Sleep time, summer. Indoor 76F is warm. Cool to 75F." — is the output of ~100 lines of Python analysis compressed into one sentence. Python already figured out the time period, selected the comfort range, checked the forecast, analyzed whether pre-cooling is needed, and routed the user message. The SLM gets a clear, unambiguous instruction and just needs to format the final decision.

**This is the core architectural insight of the project:** Python does 90% of the thinking. The SLM does the last 10% — interpreting nuance, handling natural language edge cases, and generating a human-readable reasoning string. Each component does what it's good at. Python is good at deterministic logic. SLMs are good at language understanding. Neither is asked to do the other's job.

If I had a 70B model, I might have been lazy — dumped everything into the prompt and hoped the model would figure it out. The hardware constraint forced a cleaner architecture.

### 3. Telegram as the Only Interface

**Decision:** No web dashboard. No mobile app. Telegram bot only.

**Why:** Telegram is free, already on both our phones, supports rich messages, and has a mature Python SDK. Building a dashboard would have taken longer than the entire agent — and for what? Two users don't need a dashboard. We need a conversation.

The deeper insight: the right interface for a 2-person household is *chat*, not a control panel. My wife doesn't want to check a dashboard. She wants to say "make it warmer" and have it happen. Telegram gives us that with zero UI work.

### 4. 20-Minute Evaluation Cycle

**Decision:** The agent evaluates every 20 minutes, not continuously.

**Why:** This comes from lived experience, not theory. In years of manually adjusting our thermostat, I've never needed to change the setting more than once in 30 minutes. HVAC systems are physically slow — it takes 10-15 minutes for a temperature change to register on the thermostat sensor. Evaluating every 5 minutes would just produce 3 identical "no_change" decisions.

The 20-minute cadence also respects the climate reality of Brentwood, CA. We don't get sudden 15-degree temperature swings. Weather changes are gradual and predictable. A 20-minute window captures every meaningful shift without wasting compute.

When a user sends a Telegram message, the agent runs an immediate evaluation cycle regardless of the timer. So real-time responsiveness is preserved for the cases that matter — human requests.

### 5. Open-Meteo over OpenWeatherMap

**Decision:** Switched from OpenWeatherMap (OWM) to Open-Meteo as the primary weather source on April 6.

**Why:** Open-Meteo gives better data, not just free data.

| Capability | Open-Meteo | OWM Free Tier |
|---|---|---|
| True daily high/low | Yes (meteorological model) | No (must estimate from samples) |
| Forecast resolution | Hourly (48 data points) | 3-hour intervals (fewer points) |
| API key required | No | Yes |
| Cost | Free, no limits | Free tier, 1,000 calls/day |

The critical difference is **true daily high/low**. OWM's free tier doesn't provide daily aggregate forecasts — you get 3-hour snapshots and have to calculate max/min yourself. On April 6, I caught OWM reporting a forecast high of 71F when the actual daily high was 79F. That's an 8-degree error that directly affects HVAC mode switching (COOL vs HEAT). Open-Meteo's meteorological model gives the real daily extremes.

OWM is kept as a fallback. If Open-Meteo is down, the system degrades to OWM rather than flying blind.

### 6. Hard Guardrails the AI Cannot Override

**Decision:** Temperature bounds (65-80F), rate limits (6 changes/hour), and manual override backoff (2 hours) are enforced in Python, not by the LLM.

**Why:** I don't trust a 2B parameter model to never hallucinate "set_temperature: 45" on a cold night. The LLM is an untrusted component. It proposes; Python disposes.

This is the single most important architectural decision in the project. Every AI product needs a "the AI is wrong" layer. Mine clamps temperatures, blocks excessive changes, and respects when someone physically adjusts the thermostat. The LLM never touches the Nest API directly.

| Guardrail | Value | Why |
|---|---|---|
| Min temperature | 65F | Pipe freeze prevention, basic comfort floor |
| Max temperature | 80F | Energy waste prevention, equipment protection |
| Max changes/hour | 6 | Prevents HVAC short-cycling (compressor damage) |
| Manual override backoff | 120 min | If someone walks to the thermostat, they meant it |
| User request backoff | 40 min | Don't immediately overrule what the user asked for |

User requests from Telegram bypass rate limits and override backoff. The hierarchy is clear: **human > guardrails > AI**.

### 7. Conversation over Dashboard (What "No Dashboard" Really Means)

This wasn't laziness — it was a deliberate product choice. A dashboard is a *pull* interface: the user has to go look at it. A conversation is a *push* interface: the agent tells you what it did and why.

The "whoa" moment doesn't happen in a dashboard. It happens when the agent texts you: "Pre-cooling to 75F — forecast shows 98F peak in 2 hours. Better to cool now while it's still 88F outside." That's the product. That's the intelligence made visible.

---

## Decisions I Explicitly Did Not Make

What you *don't* build defines a product as much as what you do.

| Scoped Out | Why |
|---|---|
| Web dashboard | Two users. Chat is the right interface. Dashboard adds complexity with zero user value for our household. |
| Occupancy detection | Our schedule is predictable enough. Sensors add hardware cost and integration complexity for marginal improvement. |
| Cloud deployment | Violates the privacy constraint. Also adds recurring cost (the thing I'm trying to avoid). |
| Multi-user preferences | My wife and I have similar comfort ranges. Personalization per user is over-engineering for two people who agree on temperature. |
| Voice assistant integration | Would require cloud (Alexa/Google) or complex local setup. Telegram is already on our phones and faster than talking to a speaker. |
| Energy cost tracking | SDM API doesn't expose energy telemetry for our Nest model. Can't track what I can't measure. Deferred until data is available. |

---

## Pivots & What Drove Them

### Qwen 4B to Gemma 4 E2B: Same Accuracy, Half the Size

**What happened:** I started with Qwen 4B (the model I had on hand). It worked, but I wanted to know if I could run something lighter without sacrificing decision quality.

**What I did:** I built a dedicated test harness (`testHarness/test_zone_routing.py`) with 59 test scenarios across 9 categories — explicit zone commands, ambiguous requests, relative adjustments, contradictory inputs, guardrail enforcement, and natural language. Each model was tested identically.

**Results:**

| Metric | Qwen 4B | Gemma 4 E2B |
|---|---|---|
| Exact match (PASS) | 109/118 (92.4%) | 109/118 (92.4%) |
| Acceptable (PASS + CLOSE) | 112/118 (94.9%) | 112/118 (94.9%) |
| JSON parse errors | 0 | 0 |
| Categories at 100% | 5/9 | 5/9 |

**Identical.** Same pass rate, same failure patterns, same edge cases. Both models struggle with the same things (complex context like "I'm working from home today" and contradictory statements). Neither is better.

**Decision:** Ship Gemma 4 E2B. It's lighter on VRAM, and since accuracy is identical, there's no reason to run the larger model.

**PM lesson:** Don't assume bigger = better. Benchmark before you decide. The test harness took 2 hours to build and saved me from running a needlessly large model 24/7.

### OpenWeatherMap to Open-Meteo: Data Quality > Brand Recognition

Covered above in Key Decisions. The trigger was a specific incident: OWM reported 71F forecast high, reality was 79F. That 8-degree gap caused the system to stay in HEAT mode when it should have been in COOL. Open-Meteo's true daily high/low fixed it immediately.

### Time-Based to Cycle-Based Message Expiry

**Before:** User messages expired after 2 hours (wall clock time).

**Problem:** If the agent evaluated at minute 0 and the user sent a message at minute 1, the message would be active for almost 6 evaluation cycles (119 minutes). That's 5 extra cycles where the LLM is still trying to follow a stale instruction.

**After:** Messages expire after 2 evaluation cycles regardless of clock time. This means a message is active for exactly 2 decisions (the immediate response + one follow-up), then the agent returns to autonomous mode.

### False Manual Override Detections

**Problem:** The agent was detecting "manual overrides" that never happened. Two root causes:

1. **Fahrenheit-Celsius rounding.** The Nest API stores temperatures in Celsius. Setting 75F = 23.89C, which the API rounds to 24C = 75.2F. On the next read, the agent sees 75.2F vs its expected 75F, thinks someone manually changed it, and triggers a 2-hour backoff. The fix: compare with a 1.5F tolerance.

2. **Mode switching.** When the agent switches from COOL to HEAT, the Nest's target temperature changes (cool setpoint vs heat setpoint). The agent saw this as a manual override. The fix: suppress override detection immediately after a mode switch.

**PM lesson:** Physical systems have rounding, latency, and state transitions that software doesn't. Every "simple" integration with hardware has hidden edge cases.

---

## What I Learned About AI Product Design

### 1. The LLM is the Least Important Part of the System

This sounds counterintuitive for an "AI thermostat," but the LLM is just the final decision-maker in a pipeline of Python logic that does the real work:

- Python determines the time period (sleep/awake/winding down)
- Python selects the comfort range (summer vs winter)
- Python analyzes the forecast (pre-cool? pre-heat? heatwave?)
- Python routes user messages to the correct zone
- Python builds a 1-2 sentence directive ("Sleep time, summer. Indoor 76F is warm. Cool to 75F.")
- **The LLM just says yes or adjusts slightly.**

This is the most transferable insight: **don't ask the AI to be smart. Make the system smart and ask the AI to confirm.** A 2B parameter SLM can reliably say "yes, set to 75" when you've already done the reasoning in Python and handed it a 200-token prompt. It cannot reliably analyze raw forecast data, cross-reference user preferences, and determine the optimal temperature from scratch in a 4096-token context window. The hardware constraint (crappy GPU, small model, tight context) forced this separation — and it's a better architecture than what I'd have built with unlimited compute.

### 2. Prompt Engineering is Product Design

The system prompt isn't an engineering artifact — it's a product spec. Every word in the directive shapes the user experience:

- "Comfort range is a GUIDE, not a hard rule" → the agent doesn't robotically snap to 75F every cycle
- "Energy-saving bias: prefer no_change" → the agent doesn't thrash the HVAC
- "User said: [message]. Follow their request exactly." → user intent is never reinterpreted

When I changed "Follow the user's temperature preference" to "Follow their request exactly," false interpretations dropped noticeably. That's a product decision made in a prompt.

### 3. JSON Reliability is a Solved Problem (With the Right Approach)

Both Qwen 4B and Gemma 4 E2B achieved **0% JSON parse errors** across 118 test cases. The key:

- Fixed schema in the prompt (no flexibility)
- Explicit "Respond ONLY with valid JSON" instruction
- Retry once on parse failure with stronger instruction
- Strip markdown code fences (Gemma wraps JSON in ` ```json ``` ` blocks)

At the 2-4B parameter scale, structured output works if you constrain it enough. The models aren't unreliable — the prompts are usually too permissive.

### 4. A Test Harness is the Most Important Thing You Build

Unit tests validate your Python logic. The test harness validates your *AI product*.

I built a standalone test harness (`testHarness/test_zone_routing.py`) with 59 real-world scenarios across 9 categories: explicit zone commands ("set bedroom to 75"), ambiguous requests ("make it cooler"), relative adjustments ("drop it by 2 degrees"), contradictory inputs ("make it colder, set to 80"), and natural language ("it feels stuffy in here"). It runs each scenario against the actual LLM with the actual prompt, scores PASS/CLOSE/FAIL, and produces a report.

**Why this matters more than unit tests:**

Unit tests told me `validate_response()` correctly rejects malformed JSON. The test harness told me that Qwen 4B interprets "I'm working from home today" as a temperature request 25% of the time. One tests code. The other tests the product.

The test harness enabled three things that would have been impossible without it:

1. **Model comparison with confidence.** When I wanted to switch from Qwen 4B to Gemma 4 E2B, I didn't guess — I ran both models through identical 118-check benchmarks. Both scored 94.9%. That's not "it seems to work." That's a quantified, reproducible decision.

2. **Prompt iteration with a safety net.** Every time I changed the directive format or the zone routing rules, I re-ran the harness. Version 1 scored 90.0%. The ultra-lean 4-rule system scored 94.1%. Adding semantic aliases pushed it to 94.9%. Each change was measured, not vibed.

3. **Regression detection.** When I simplified the prompt for SLM context constraints, the harness caught that "contradictory" scenarios dropped from 75% to 50%. I knew exactly what I was trading off.

**The evolution the harness tracked:**

| Version | Prompt Approach | Acceptable Rate |
|---|---|---|
| v1 | Basic zone routing | 90.0% |
| v2 | 3-rule priority + eco handling | 88.1% (regression — caught it) |
| v3 | Ultra-lean 4-rule system | 94.1% |
| v4 | 4-rule + semantic aliases | 94.9% |

Without the harness, I'd be shipping prompt changes blind. With it, every change is a measured experiment. For any AI product — especially one built on SLMs where prompt wording is load-bearing — a test harness isn't optional tooling. It's the product's quality gate.

### 5. Production AI Needs a "Respect the Human" Layer

Three features that turned out to be essential:

- **Manual override backoff:** If someone physically adjusts the thermostat, the agent backs off for 2 hours. This prevents the maddening experience of adjusting your thermostat and having it change back 20 minutes later.
- **User request priority:** Telegram messages bypass all rate limits and override backoff. When a human speaks, the agent listens immediately.
- **Message expiry:** After 2 cycles, the agent stops following stale instructions and returns to autonomous mode. Without this, a "set it to 72" from Tuesday morning would still be influencing decisions on Tuesday night.

---

## System in Production: By the Numbers

| Metric | Value |
|---|---|
| Days running in production | 19 (March 31 — April 18, 2026) |
| Zones managed | 2 (Upstairs Bedroom, Downstairs Kitchen) |
| Evaluation cycle | Every 20 minutes |
| Autonomous decisions per day | ~144 (72 per zone) |
| Total decisions (estimated) | ~2,700+ |
| Safety incidents | 0 |
| JSON parse failures in production | 0 |
| QA bugs found and fixed | 4 (all on day 1, all with regression tests) |
| Unit tests | 49 (all passing) |
| Benchmark test scenarios | 59 scenarios, 118 checks |
| Model accuracy (Gemma 4 E2B) | 94.9% acceptable |
| Cloud AI cost | $0.00 |
| Total project cost | $5.00 (Google SDM one-time fee) |

---

## What I'd Do Differently

1. **Build the test harness first, not after.** I built it to compare Qwen vs Gemma, but I should have built it before writing any prompt logic. Would have caught the zone-routing edge cases earlier.

2. **Start with Open-Meteo.** I defaulted to OpenWeatherMap because it's the most well-known weather API. That's brand bias, not product thinking. Open-Meteo is objectively better for this use case (true daily extremes, hourly resolution, no API key).

3. **Design for the physical world from day one.** The Celsius rounding bug, the mode-switch false override, the HVAC short-cycling risk — these are all "obvious" in hindsight but invisible when you're thinking in software abstractions. Physical systems deserve their own test category.

4. **Track energy savings from the start.** My design doc lists "HVAC runtime comparison" as a success criterion, but I didn't build the baseline measurement. Now I have 19 days of agent data but no pre-agent comparison. The most compelling metric for this product — "saved X% on energy" — is the one I can't prove.

---

## Architecture (One Picture)

```
User (Telegram)
    │
    ▼
┌─────────────────────────────────────────────────┐
│              AGENT CORE (Python)                │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ Directive │  │ Guardrail│  │ Decision      │ │
│  │ Builder   │─▶│ Checker  │─▶│ Executor      │ │
│  │ (Python)  │  │ (Python) │  │ (Nest API)    │ │
│  └────┬─────┘  └──────────┘  └───────────────┘ │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐                                   │
│  │ Local LLM│  Gemma 4 E2B via llama.cpp       │
│  │ (on GPU) │  Starts per cycle, stops after    │
│  └──────────┘                                   │
│                                                  │
│  Data Sources:                                  │
│  ├── Nest SDM API (indoor temp, humidity, mode) │
│  ├── Open-Meteo (forecast, daily high/low)      │
│  ├── SQLite (decisions, messages, climate log)  │
│  └── Telegram (user messages)                   │
└─────────────────────────────────────────────────┘
    │
    ▼
Nest Thermostats (2 zones)
```

---

## Success Criteria Revisited

From the original design doc (March 29, 2026):

| Criterion | Status | Evidence |
|---|---|---|
| Agent runs 24+ hours without intervention | **Met** | Running 19 days autonomously |
| Wife can text naturally and get sensible responses | **Met** | Two whitelisted users active on Telegram |
| HVAC runtime comparison vs manual baseline | **Not measured** | No pre-agent baseline was logged (see "What I'd Do Differently") |
| At least one friend says "whoa" | **Met** | The Telegram reasoning display is the product's best demo |
| Agent explains reasoning before every change | **Met** | Every `set_temperature` decision includes LLM reasoning in the Telegram notification |
