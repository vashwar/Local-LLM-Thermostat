# AI Thermostat Agent - Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI THERMOSTAT SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │      EXTERNAL SERVICES               │
                    ├──────────────────────────────────────┤
                    │ • Google Nest SDM API                │
                    │ • OpenWeatherMap API                 │
                    │ • Telegram Bot API                   │
                    │ • llama.cpp Server (Local LLM)       │
                    └──────────────────────────────────────┘
                                    ▲
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌──────────────────┐  ┌──────────┐  ┌──────────────┐
        │   Nest API       │  │ Weather  │  │ Telegram Bot │
        │   Module         │  │ Module   │  │ Module       │
        └──────────────────┘  └──────────┘  └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │       AGENT CORE                     │
                    │  (Main Logic & Decision Engine)      │
                    ├──────────────────────────────────────┤
                    │ • Evaluation Loop (20-min interval)  │
                    │ • Context Builder                    │
                    │ • LLM Caller                         │
                    │ • Guardrail Checker                  │
                    │ • Decision Executor                  │
                    └──────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │       DATABASE (SQLite)              │
                    │                                      │
                    ├──────────────────────────────────────┤
                    │ • climate_log (training data)        │
                    │ • decisions (90-day retention)       │
                    │ • messages (90-day retention)        │
                    │ • errors (90-day retention)          │
                    │ • heartbeat                          │
                    └──────────────────────────────────────┘
```

---

## Execution Flow - Single Evaluation Cycle

```
┌─ START EVALUATION CYCLE ─────────────────────────────────────────┐
│                                                                    │
│  1. FETCH INDOOR STATE                                           │
│     ├─> Nest API (per zone)                                     │
│     ├─> Indoor temp, humidity, mode, target, HVAC status        │
│     └─> ThermostatState objects                                 │
│                                                                    │
│  2. FETCH OUTDOOR STATE                                         │
│     ├─> OpenWeatherMap (current + 5-day forecast)              │
│     ├─> Cache with 20-min freshness, 6-hour stale detection     │
│     └─> WeatherData object                                      │
│                                                                    │
│  3. CHECK FOR USER MESSAGES                                     │
│     ├─> Query database.messages (recent)                        │
│     ├─> Detect natural language commands                        │
│     └─> Set _user_triggered flag if present                     │
│                                                                    │
│  4. DETECT MANUAL OVERRIDES                                     │
│     ├─> Compare current target vs. last known                   │
│     ├─> Log override + trigger 120-min backoff                  │
│     └─> Send Telegram alert                                     │
│                                                                    │
│  5. START LLM SERVER                                            │
│     ├─> llama-server starts on port 8080 (or configured)       │
│     ├─> Wait up to 60 seconds for readiness                     │
│     └─> Health check via /health endpoint                       │
│                                                                    │
│  6. FOR EACH ZONE (Multi-zone support):                        │
│     │                                                             │
│     ├─ 6a. BUILD CONTEXT                                        │
│     │      ├─> Indoor state + outdoor state + user messages     │
│     │      ├─> Python-driven directive analysis                 │
│     │      │   (comfort ranges, time period, forecast)          │
│     │      └─> System prompt with all context                   │
│     │                                                             │
│     ├─ 6b. CALL LOCAL LLM                                       │
│     │      ├─> POST /v1/chat/completions (llama.cpp compatible) │
│     │      ├─> Request JSON decision response                   │
│     │      ├─> Retry once on JSON decode failure                │
│     │      └─> Return: (response_text, is_valid_json)           │
│     │                                                             │
│     ├─ 6c. VALIDATE RESPONSE                                    │
│     │      ├─> Check JSON structure                             │
│     │      ├─> Verify action ("set_temperature" or "no_change") │
│     │      ├─> Check temperature in range [65F, 80F]            │
│     │      └─> Extract: {action, temperature, reasoning, msg}   │
│     │                                                             │
│     ├─ 6d. CHECK GUARDRAILS                                     │
│     │      ├─> Rate limit: max 6 changes/hour (unless user)    │
│     │      ├─> Manual override backoff: 120 minutes             │
│     │      ├─> Temp bounds: 65F-80F (hard-coded)                │
│     │      └─> Skip all checks if user_triggered=true           │
│     │                                                             │
│     └─ 6e. EXECUTE DECISION                                     │
│            ├─> If "set_temperature": call Nest API              │
│            ├─> Log to climate_log + decisions                   │
│            ├─> Log to database (decision tracking)              │
│            └─> Send Telegram notification (if changed)          │
│                                                                    │
│  7. STOP LLM SERVER                                             │
│     ├─> Terminate process                                       │
│     └─> Free GPU memory                                         │
│                                                                    │
│  8. POST-CYCLE TASKS                                            │
│     ├─> Update heartbeat timestamp                              │
│     ├─> Check if Sunday → send weekly report                   │
│     ├─> Daily cleanup (delete records >90 days)                 │
│     └─> Send summary to Telegram                                │
│                                                                    │
└─ END EVALUATION CYCLE ──────────────────────────────────────────┘
```

---

## Component Architecture

### 1. **Agent Core** (`agent.py`)
```
agent.py
├── Module State
│   ├── _config (YAML configuration)
│   ├── _last_evaluation_result
│   ├── _evaluation_counter
│   ├── _user_triggered
│   └── _telegram_send_fn
│
├── Functions
│   ├── load_config() → dict
│   ├── call_llm(system_prompt) → (response, is_json)
│   ├── validate_response(response_text) → (valid, error_msg)
│   ├── build_context(thermo_state, weather) → system_prompt
│   ├── _build_directive(...) → directive_text
│   ├── check_guardrails(decision) → (allowed, reason)
│   ├── execute_decision(decision, thermo_state) → action
│   ├── run_evaluation_cycle() → decision
│   └── agent_loop() → [infinite async loop]
│
└── Constants
    ├── TEMP_MIN = 65F
    ├── TEMP_MAX = 80F
    ├── MAX_CHANGES_PER_HOUR = 6
    ├── MANUAL_OVERRIDE_BACKOFF_MINUTES = 120
    └── SYSTEM_PROMPT = [Qwen 4B instruction]
```

### 2. **Nest API Wrapper** (`nest_api.py`)
```
nest_api.py
├── State
│   ├── _tokens (OAuth tokens, auto-refresh)
│   ├── _devices (device list from config)
│   └── _last_known_targets (for override detection)
│
├── Data Classes
│   └── ThermostatState
│       ├── name, device_id, indoor_temp, humidity
│       ├── mode, target_temp, hvac_running
│
├── Functions
│   ├── init_nest(tokens_path, devices) → void
│   ├── get_all_thermostat_states() → [ThermostatState]
│   ├── get_thermostat_state(device_id) → ThermostatState
│   ├── set_temperature(temp_f, device_id) → bool
│   ├── set_mode(mode, device_id) → bool
│   ├── detect_manual_override(device_id) → bool
│   └── _refresh_token() → void
│
└── API Details
    ├── Endpoint: smartdevicemanagement.googleapis.com/v1
    ├── Auth: OAuth 2.0 (auto-refresh on 401)
    ├── Conversion: Celsius ↔ Fahrenheit
    └── Per-device: cool/heat setpoint management
```

### 3. **Weather Module** (`weather.py`)
```
weather.py
├── State
│   ├── _api_key, _lat, _lon
│   ├── _cache_minutes (default: 20)
│   ├── _stale_hours (default: 6)
│   ├── _cached_weather
│   └── _cache_timestamp
│
├── Data Classes
│   └── WeatherData
│       ├── current_temp (F), humidity (%)
│       ├── forecast_summary (1-2 sentences)
│       ├── is_stale, alerts
│
├── Functions
│   ├── init_weather(api_key, lat, lon, cache_min, stale_hr) → void
│   ├── get_weather() → WeatherData [cached]
│   ├── get_forecast_analysis() → ForecastAnalysis
│   ├── check_weather_alerts(weather_data) → [alerts]
│   ├── _fetch_current() → current_data
│   └── _fetch_forecast() → forecast_data
│
└── API Details
    ├── Endpoint: api.openweathermap.org
    ├── Free tier: 1000 calls/day (uses ~72/day)
    ├── Data: current temp, 5-day forecast
    └── Alerts: heat, cold, extreme weather
```

### 4. **LLM Server Manager** (`llm_server.py`)
```
llm_server.py
├── State
│   ├── _process (subprocess handle)
│   ├── _server_exe (path to llama-server.exe)
│   ├── _model_path (path to Qwen 4B GGUF)
│   ├── _port (default: 8080)
│   └── _health_url
│
├── Functions
│   ├── init(port) → void
│   ├── is_running() → bool
│   ├── start() → bool [waits up to 60s]
│   └── stop() → void
│
└── Details
    ├── Model: Qwen 4B (3GB VRAM)
    ├── Context: 4096 tokens
    ├── GPU layers: 99 (full offload)
    ├── Startup: 2-20 seconds
    └── API: OpenAI /v1/chat/completions compatible
```

### 5. **Telegram Bot** (`telegram_bot.py`)
```
telegram_bot.py
├── State
│   ├── _app (telegram.ext.Application)
│   ├── _config, _whitelisted_ids
│
├── Commands
│   ├── /status → Show current state + last decision
│   ├── /history → Show last 2 decisions
│   ├── /export → CSV download of climate log
│   └── [Any text message] → Natural language request
│
├── Handlers
│   ├── cmd_status(update, context) → async
│   ├── cmd_history(update, context) → async
│   ├── cmd_export(update, context) → async
│   ├── handle_message(update, context) → async [triggers evaluation]
│   └── _safe_reply(message, text, retries) → async
│
├── Integration
│   ├── Calls agent.set_telegram_send_fn(async_send)
│   ├── Calls agent.trigger_evaluation() on message
│   └── Calls database functions for retrieval
│
└── Security
    ├── Whitelist: configured chat IDs only
    ├── Polling: long-polling (not webhooks)
    └── Retry: exponential backoff on timeout
```

### 6. **Database** (`database.py`)
```
database.py
├── Tables (SQLite)
│   ├── climate_log (training data, indefinite retention)
│   │   └── timestamp, zone, indoor_temp, humidity, outdoor_temp,
│   │       forecast, hvac_mode, hvac_running, target_temp, action
│   │
│   ├── decisions (90-day retention)
│   │   └── timestamp, zone, indoor_temp, outdoor_temp,
│   │       action, temperature, reasoning, raw_response
│   │
│   ├── messages (90-day retention)
│   │   └── timestamp, chat_id, text, agent_response
│   │
│   ├── errors (90-day retention)
│   │   └── timestamp, module, error_type, details
│   │
│   └── heartbeat (single row)
│       └── timestamp (last evaluation)
│
├── Functions
│   ├── init_db(db_path) → void
│   ├── log_decision(...) → void
│   ├── log_climate(...) → void
│   ├── log_message(...) → void
│   ├── log_error(...) → void
│   ├── get_recent_messages(limit) → [messages]
│   ├── get_climate_log_since(datetime) → [entries]
│   ├── count_temp_changes_last_hour() → int
│   ├── get_last_manual_override_time() → datetime
│   ├── cleanup_old_records() → void (daily)
│   └── update_heartbeat() → void
│
└── Optimizations
    ├── WAL mode for concurrent access
    ├── Indexes on timestamp
    ├── Busy timeout: 5 seconds
    └── Row factory for dict-like access
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INCOMING DATA FLOWS                                  │
└─────────────────────────────────────────────────────────────────────────────┘

USER (Telegram)              EXTERNAL APIS                  HARDWARE
    │                            │                              │
    │                            │                              │
    ├──> Message                 ├──> Nest API ──────────────> Thermostat
    │    (trigger)               │    (get state)              (read temp)
    │                            │                              │
    ├──> Command                 ├──> Nest API ──────────────> Thermostat
    │    (/status,               │    (set target)             (write temp)
    │     /history,              │                              │
    │     /export)               ├──> OpenWeatherMap ─────────> Weather
    │                            │    (current + forecast)      Data
    │                            │                              │
    └────────────────────────────┴──────────────────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │   AGENT CORE             │
    │  (decision engine)       │
    └──────────────────────────┘
             │
             ├─────────────────────────┬─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │ Validate against │    │ Check Guardrails │    │ Log to Database  │
    │ Safety Bounds    │    │  (rate limits,   │    │  (climate_log,   │
    │  (65-80F)        │    │   overrides)     │    │   decisions)     │
    └────────┬─────────┘    └────────┬─────────┘    └──────────────────┘
             │                      │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │  Execute Decision    │
             │  (if approved)       │
             └──────────┬───────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌──────────┐
    │ Nest API│   │ Database │   │ Telegram │
    │(set     │   │(log)     │   │(notify)  │
    │ temp)   │   │          │   │          │
    └─────────┘   └──────────┘   └──────────┘
         │              │              │
         ▼              ▼              ▼
      Thermostat    SQLite DB      User Phone
```

---

## Multi-Zone Support

```
Single Evaluation Cycle (affects N zones)

Agent runs these steps ONCE:
├─ 1. Fetch all thermostat states (all zones)
├─ 2. Fetch weather (shared)
├─ 3. Fetch user messages (shared)
├─ 4. Start LLM server (once)
│
└─ FOR EACH ZONE (parallel per zone, serial for LLM):
   ├─ Build context (includes other zones' state for reference)
   ├─ Call LLM with zone-specific prompt
   ├─ Validate response
   ├─ Check guardrails
   ├─ Execute decision (Nest API call per zone)
   └─ Log to database
        └─ Finally: Stop LLM server (once)

Example: 2 zones (Upstairs + Downstairs)
├─ Evaluation 1: Upstairs
│  └─ Context includes: Upstairs state, Downstairs state, weather
├─ Evaluation 2: Downstairs
│  └─ Context includes: Downstairs state, Upstairs state, weather
└─ Both decide independently, both can change temp or hold
```

---

## Safety Guardrails (Hard-Coded)

```
┌─────────────────────────────────────────────────────────────────┐
│ GUARDRAILS (Cannot be overridden by AI or config)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. TEMPERATURE BOUNDS                                           │
│    └─ Minimum: 65F, Maximum: 80F                               │
│       (LLM cannot request outside this range)                  │
│                                                                  │
│ 2. RATE LIMITING (automatic cycles only)                        │
│    └─ Max 6 temperature changes per hour                       │
│       (User-triggered evaluations bypass this)                 │
│                                                                  │
│ 3. MANUAL OVERRIDE BACKOFF (automatic cycles only)              │
│    └─ 120 minutes after manual change detected                 │
│       (User-triggered evaluations bypass this)                 │
│       (System detects: current_target vs. last_known)          │
│                                                                  │
│ 4. LLM RESPONSE VALIDATION                                      │
│    ├─ Must be valid JSON                                       │
│    ├─ Must have "action" field (set_temperature | no_change)   │
│    ├─ Must have "reasoning" field                              │
│    ├─ If set_temperature: must have temperature ∈ bounds       │
│    └─ Malformed responses are rejected + logged                │
│                                                                  │
│ 5. JSON PARSING RETRIES                                        │
│    └─ Retry once if LLM response is not valid JSON             │
│       (second failure aborts and logs error)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Async Concurrency Model

```
Main Entry Point: asyncio.run(main())
│
├─ Coroutine 1: agent_loop()
│  ├─ Infinite loop, runs every 20 minutes (configurable)
│  ├─ Calls run_evaluation_cycle() [can block up to 5min]
│  ├─ Awaits result with timeout=300 seconds
│  ├─ Checks for _user_triggered flag (woken on demand)
│  └─ Sends weekly report (Sundays), daily cleanup
│
└─ Coroutine 2: telegram_bot.start_bot()
   ├─ Long-polling (not webhooks)
   ├─ Receives updates in real-time
   ├─ Calls cmd_* handlers asynchronously
   ├─ Calls agent.trigger_evaluation() on messages
   └─ Awaits replies with retry logic
        │
        └─ Wakes agent_loop via _user_triggered flag
           (Agent doesn't wait for next interval)

Both run concurrently via asyncio.gather()
```

---

## Configuration File (`config.yaml`)

```yaml
llm:
  server_exe: "C:\\path\\to\\llama-server.exe"         # Full path to llama-server executable
  model_path: "C:\\path\\to\\Qwen3-4B-Instruct.gguf"   # Full path to model GGUF file
  endpoint: "http://localhost:8080/v1/chat/completions"
  model: "qwen4b"
  temperature: 0.3        # Low = more deterministic
  top_p: 0.9              # Nucleus sampling
  max_tokens: 500
  timeout_seconds: 120

nest:
  tokens_path: "nest_tokens.json"
  devices:
    - name: "Upstairs Bedroom"
      device_id: "enterprises/PROJECT_ID/devices/DEVICE_ID"
    - name: "Downstairs Kitchen"
      device_id: "enterprises/PROJECT_ID/devices/DEVICE_ID"

weather:
  api_key: "YOUR_KEY"
  latitude: 33.0198
  longitude: -96.6989
  cache_minutes: 20       # Fresh data window
  stale_hours: 6          # "STALE" if older

telegram:
  bot_token: "YOUR_BOT_TOKEN"
  whitelisted_chat_ids:
    - 123456789           # Your chat ID
    - 987654321           # Family member

comfort:
  summer_range: [75, 80]  # Cooling season target
  winter_range: [68, 72]  # Heating season target
  user_request_hours: 2   # Honor requests this long
  pre_heat_minutes: 30    # Wake up early to pre-heat
  sleep_cool_temp: 75     # Pre-cool to this before sleep
  sleep_cool_override_temp: 80  # Unless outdoor > this

schedule:
  sleep_time: "22:00"
  wake_time: "06:30"
  sleep_temp: 68
  wake_temp: 72
  away_temp: 78
  home_temp: 72

agent:
  loop_interval_minutes: 20
  db_path: "thermostat.db"
  log_level: "INFO"
```

---

## Decision Logic Flow (Python-Driven)

```
For each zone's evaluation:

1. DETERMINE TIME PERIOD (Python analysis)
   ├─ sleep (after 22:00 or before 06:30-30min)
   ├─ pre_wake (06:00-06:30)
   ├─ waking_up (06:30-07:30)
   ├─ work (9:00-17:00 on weekdays)
   ├─ winding_down (21:00+)
   └─ home (default)

2. COMFORT RANGE SELECTION (based on mode + season)
   ├─ HEAT mode → winter_range [68, 72]
   └─ COOL mode → summer_range [75, 80]

3. ANALYZE SITUATION (Python pre-processor)
   ├─ Indoor temp vs. comfort range
   ├─ Outdoor temp + forecast
   ├─ Time period + schedule awareness
   ├─ Recent user messages (highest priority)
   ├─ Manual override detection
   └─ Weather alerts (extreme heat/cold)

4. BUILD DIRECTIVE (1-2 sentences for LLM)
   Examples:
   ├─ "Sleep time, summer. Indoor 76F is warm. Cool to 75F."
   ├─ "Work hours. Wider range OK to save energy."
   ├─ "User said: 'set upstairs to 78'. Follow exactly."
   └─ "Extreme heat outside (105F). Pre-cool aggressively."

5. LLM DECISION
   ├─ INPUT: System prompt + directive + user messages
   ├─ OUTPUT: JSON {action, temperature, reasoning, message_to_user}
   └─ NOTE: LLM makes final call, Python provides the reasoning

6. GUARDRAILS ENFORCEMENT
   ├─ Check temp bounds
   ├─ Check rate limits
   ├─ Check override backoff
   └─ If blocked: log "BLOCKED:action" and skip execution

7. EXECUTION
   ├─ If "set_temperature": call Nest API
   ├─ If "no_change": skip API call
   └─ Log decision + climate snapshot to database
```

---

## Startup Sequence

```
python start.bat
    │
    ▼
python agent.py
    │
    ├─ Load config from config.yaml
    ├─ Setup logging (INFO level)
    │
    ├─ database.init_db()
    │  └─ Create SQLite tables if needed (climate_log, decisions, etc.)
    │
    ├─ weather.init_weather()
    │  └─ Store API key, lat/lon, cache parameters
    │
    ├─ nest_api.init_nest()
    │  └─ Load OAuth tokens from nest_tokens.json
    │
    ├─ llm_server.init()
    │  └─ Set port to 8080 (from config)
    │
    ├─ telegram_bot.init_bot()
    │  └─ Set whitelist IDs, create Application
    │
    ├─ agent.load_config()
    │  └─ Load YAML (called again by main)
    │
    └─ asyncio.gather(
        agent_loop(),
        telegram_bot.start_bot()
    )
       ├─ agent_loop() starts waiting for Telegram bot
       └─ telegram_bot.start_bot() registers send function
           └─ agent.set_telegram_send_fn(async_send)
               └─ agent_loop() wakes and starts 20-min cycles
```

---

## Error Handling & Logging

```
Error Scenarios:

1. LLM Server Failure
   └─ log_error("llm_server", "start_failed", "...")
   └─ Skip evaluation cycle
   └─ Retry next 20-min interval

2. Nest API 401 Unauthorized
   └─ Auto-refresh token
   └─ Retry request
   └─ If still fails: log_error("nest_api", ...)

3. Invalid JSON from LLM
   └─ log_error("llm", "invalid_json", response[:500])
   └─ Retry once
   └─ If still invalid: skip zone

4. Validation Failed
   └─ log_error("llm", "validation_failed", error_msg)
   └─ Skip zone

5. Weather API Timeout
   └─ Return cached data if available
   └─ Mark as [STALE] if > 6 hours
   └─ Fall back to defaults

6. Telegram Timeout
   └─ Retry up to 3 times with exponential backoff
   └─ Log error if all retries fail

All errors are logged to:
├─ Console (DEBUG/INFO/WARNING/ERROR)
├─ SQLite errors table
└─ No data is lost due to error
```

---

## Performance Considerations

```
Typical Evaluation Cycle Timing:

Step                              Duration
─────────────────────────────────────────────
1. Fetch all thermostat states    ~2-3 seconds
2. Fetch weather (cached)         ~0.5 seconds
3. LLM server startup             ~10-20 seconds (first time)
                                  ~2 seconds (already running)
4. Build context (per zone)       <0.1 seconds per zone
5. LLM inference (per zone)       ~5-10 seconds per zone
6. Validate + execute (per zone)  ~1-2 seconds per zone
7. LLM server shutdown            ~2 seconds

Total per cycle (2 zones):        ~20-40 seconds (first time)
                                  ~20-30 seconds (subsequent)
                                  ~5 minutes max (with timeout)

GPU Memory:
└─ Qwen 4B GGUF: ~3GB VRAM
   Freed when server stops
   Allows other tasks during non-evaluation time

API Rate Limits:
├─ Nest API: 20 calls/min (rarely hit)
├─ OpenWeatherMap: 1000 calls/day (~72 used)
├─ Telegram: 30 msg/second per bot
└─ llama.cpp: local, unlimited

Database:
├─ Climate log growth: ~5MB per year (72 entries/day)
├─ Decisions: ~1MB per year (3-6 per evaluation)
├─ Queries are indexed on timestamp
└─ Cleanup runs daily (90-day retention)
```

---

## Security Architecture

```
Boundaries:

1. LLM TRUST BOUNDARY
   ├─ Input: Context (state + user messages + forecast)
   ├─ Output: JSON decision
   ├─ Trust level: LOW (LLM is untrusted)
   │  └─ All outputs validated against guardrails
   │  └─ Temperature bounds checked (hard-coded 65-80F)
   │  └─ No shell commands or code execution
   │  └─ JSON parsing with error handling
   └─ Attack surface: LLM prompt injection
       └─ Mitigated by: system prompt fixed + user messages validated

2. EXTERNAL API TRUST
   ├─ Nest API: Authenticated (OAuth 2.0)
   ├─ Weather API: Public key-based
   ├─ Telegram: Bot token-based
   └─ All responses parsed + validated

3. LOCAL FILE TRUST
   ├─ config.yaml: Read-only, must be pre-configured
   ├─ nest_tokens.json: Generated by setup script (one-time)
   │  └─ Contains: client_id, client_secret, access/refresh tokens
   │  └─ Stored in plain text (local machine assumed secure)
   └─ thermostat.db: SQLite (no secrets stored)

4. TELEGRAM USER TRUST
   ├─ Whitelist: Only configured chat IDs can control thermostat
   ├─ Commands: /status, /history, /export (read-only)
   ├─ Messages: Natural language → triggers evaluation
   └─ No authentication beyond whitelist ID check

5. HARDWARE TRUST
   └─ Nest thermostat: Uses HTTPS + OAuth 2.0
```

---

## Extension Points

Future enhancements could integrate at these points:

```
1. Different LLM Models
   └─ Replace llama.cpp with: vLLM, ollama, HuggingFace Inference API
   └─ Change: llm_server.py startup command + endpoint

2. Additional Integrations
   ├─ Smart home: Philips Hue, LIFX, Z-Wave
   ├─ Sensors: Room-specific CO2, air quality
   ├─ Presence: Phone geolocation, occupancy sensors
   └─ Modify: weather.py + nest_api.py for each new source

3. Advanced Analytics
   ├─ Machine learning on climate_log
   ├─ Predictive pre-cooling (before forecast heat)
   ├─ Occupancy-based adjustment
   └─ Modify: decision logic in agent.py

4. Alternative Chat Interfaces
   ├─ WhatsApp, Discord, Slack
   ├─ Web dashboard
   ├─ Mobile app
   └─ Modify: telegram_bot.py handlers

5. Cloud Deployment
   ├─ AWS Lambda + RDS for database
   ├─ Azure Container Instances for LLM
   ├─ Serverless decision engine
   └─ Modify: asyncio model + database connections
```

---

## Deployment Topology (Single Machine)

```
┌──────────────────────────────────────────────────────┐
│              Windows 11 Machine                       │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Python 3.10+                                        │
│  ├─ agent.py (main process)                         │
│  ├─ telegram_bot.py (asyncio)                       │
│  ├─ nest_api.py (module)                            │
│  ├─ weather.py (module)                             │
│  ├─ llm_server.py (launches subprocess)             │
│  ├─ database.py (SQLite)                            │
│  └─ (All async, single process)                     │
│                                                       │
│  Subprocesses:                                       │
│  └─ llama-server.exe (spawned per cycle, GPU)       │
│                                                       │
│  Files:                                              │
│  ├─ config.yaml (configuration)                     │
│  ├─ nest_tokens.json (OAuth tokens)                 │
│  ├─ thermostat.db (SQLite, persistent)              │
│  └─ logs (printed to console)                       │
│                                                       │
│  Hardware:                                           │
│  ├─ Network: Ethernet (must be connected)           │
│  └─ GPU: NVIDIA with ~3GB VRAM (for Qwen 4B)        │
│                                                       │
└──────────────────────────────────────────────────────┘
         │                    │              │
         │                    │              │
    [Internet]           [Internet]     [Internet]
         │                    │              │
         ▼                    ▼              ▼
    ┌─────────┐      ┌──────────────┐  ┌──────────┐
    │ Telegram│      │ Google Cloud │  │OpenWeather
    │ Bot API │      │ Nest API     │  │Map API
    └─────────┘      └──────────────┘  └──────────┘
```

---

## Summary

- **Brain**: Qwen 4B (local LLM via llama.cpp)
- **Inputs**: Indoor temp, outdoor weather, user commands
- **Decision Cycle**: Every 20 minutes (or on demand)
- **Safety**: Hard-coded guardrails + validation
- **Multi-Zone**: Independent decisions per thermostat
- **Logging**: SQLite (climate_log, decisions, messages, errors)
- **Interface**: Telegram bot with commands
- **Privacy**: 100% local processing (no cloud AI)
