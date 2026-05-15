# AI Thermostat Agent

I built a local AI agent that manages my home's two Nest thermostats autonomously. My wife texts "it's cold" and the system reasons about indoor temp, outdoor forecast, time of day, and HVAC mode — then explains its reasoning in plain English before adjusting the temperature. All AI runs locally on a consumer-grade GPU using a 2B parameter SLM (Gemma 4 E2B via llama.cpp). Zero cloud costs. Zero data leaving my machine. It has been managing my home's climate autonomously since March 31, 2026.

**Total project cost: $5** (Google's one-time SDM API registration fee). Replaces a $500 pair of Nest Learning Thermostats.

> For the product thinking, design decisions, and lessons learned behind this project, see [PRODUCT.md](PRODUCT.md).

## How It Works

Every 20 minutes, the agent runs an evaluation cycle. Python does 90% of the reasoning (time period, comfort range, forecast analysis, zone routing) and compresses it into a ~200-token directive. The SLM makes the final call and generates a human-readable explanation. Hard-coded guardrails in Python enforce safety before any action reaches the Nest API.

```
start.bat
  └─> python agent.py
        └─> asyncio.gather(
              agent_loop(),              # 20-min scheduled evaluation
              telegram_bot.start_bot()   # Telegram polling
            )

Single Evaluation Cycle:
  1. Read indoor temp/humidity from Nest API (per zone)
  2. Read outdoor weather + forecast from Open-Meteo
  3. Read recent Telegram messages for user requests
  4. Start llama-server (on-demand, GPU)
  5. Python pre-processes context → compact directive for SLM
  6. SLM returns JSON decision → validated against guardrails
  7. Execute temperature change via Nest API (if needed)
  8. Stop llama-server (free GPU memory)
  9. Log everything to SQLite
```

When you send a Telegram message, the agent immediately runs an extra evaluation cycle with your request as priority context.

## Features

- **Multi-zone support** — manages 2 Nest thermostats independently (Upstairs Bedroom, Downstairs Kitchen)
- **Local SLM brain** — Gemma 4 E2B (2B params) runs on a consumer GPU via llama.cpp, no cloud API needed
- **On-demand LLM** — llama-server starts only during evaluations, then shuts down to free GPU memory
- **Telegram bot** — send natural language commands ("set upstairs to 78"), get status, export data
- **User requests are priority** — the AI always follows your explicit instructions over its own logic
- **Vacation mode** — OwnTracks presence detection via MQTT; sets 85F (cool) / 60F (heat) when everyone is away
- **Safety guardrails** — hard-coded temp bounds (65-80F normal, 60-85F vacation), rate limiting, manual override detection
- **Weather-aware** — Open-Meteo primary (true daily high/low, hourly resolution), OWM fallback
- **Schedule-aware** — knows sleep/wake times, adjusts comfort ranges by season and HVAC mode
- **SLM-optimized** — Python pre-processes all reasoning into a ~200-token prompt, respecting the 4096 context window
- **Benchmarked** — 59-scenario test harness validated both Qwen 4B and Gemma 4 E2B at 94.9% accuracy
- **Climate logging** — full SQLite history for analysis and weekly reports

## Prerequisites

- **Windows 10/11** (tested on Windows 11)
- **Python 3.10+**
- **NVIDIA GPU** with enough VRAM for a 2B SLM (~2GB)
- **llama.cpp** — `llama-server.exe` compiled or downloaded
- **A GGUF model** — e.g. Gemma 4 E2B (`gemma-4-e2b-it-Q8_0.gguf`) or Qwen 4B
- **Google Nest thermostat(s)** with Smart Device Management API access
- **Telegram** account

## Setup Guide

### Step 1: Install llama.cpp

1. Download a prebuilt release from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) (pick the CUDA/Vulkan version matching your GPU)
2. Extract `llama-server.exe` to a folder on your system (e.g., `C:\llama-cpp\llama-server.exe`)
3. Download a GGUF model — Gemma 4 E2B (recommended, lighter) or Qwen 4B from [HuggingFace](https://huggingface.co/)
4. Note the full paths to both files — you'll add them to `config.yaml` in Step 5

### Step 2: Get Nest API Access

This is the most involved step. You need to set up Google Cloud credentials and the Smart Device Management (SDM) API.

#### 2a. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g. "AI Thermostat")
3. Enable the **Smart Device Management API**:
   - Go to **APIs & Services > Library**
   - Search for "Smart Device Management API"
   - Click **Enable**

#### 2b. Create OAuth Credentials

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth client ID**
3. If prompted, configure the **OAuth consent screen** first:
   - User type: **External**
   - App name: "AI Thermostat" (or anything)
   - Add your email as a test user
4. Create the OAuth client:
   - Application type: **Web application**
   - Name: "AI Thermostat"
   - Authorized redirect URIs: add `https://www.google.com`
5. Copy the **Client ID** and **Client Secret** — you'll need these

#### 2c. Create a Device Access Project

1. Go to the [Device Access Console](https://console.nest.google.com/device-access)
2. Pay the one-time $5 registration fee (required by Google)
3. Create a new project:
   - Name: "AI Thermostat"
   - OAuth Client ID: paste the client ID from step 2b
4. Copy the **Project ID** (a UUID like `1bedbe14-df66-41c8-bb1f-7a86403f1548`)

#### 2d. Authorize Your Google Account

1. Open this URL in your browser (replace `YOUR_CLIENT_ID` and `YOUR_PROJECT_ID`):

   ```
   https://nestservices.google.com/partnerconnections/YOUR_PROJECT_ID/auth?redirect_uri=https://www.google.com&access_type=offline&prompt=consent&client_id=YOUR_CLIENT_ID&response_type=code&scope=https://www.googleapis.com/auth/sdm.service
   ```

2. Sign in with the Google account linked to your Nest thermostat
3. Grant access to your home and devices
4. You'll be redirected to `https://www.google.com?code=AUTHORIZATION_CODE`
5. Copy the **authorization code** from the URL (everything after `code=` up to the next `&`)

#### 2e. Exchange the Code for Tokens

Run the setup script:

```bash
python nest_setup.py
```

It will ask for:
- OAuth Client ID (from step 2b)
- OAuth Client Secret (from step 2b)
- Authorization Code (from step 2d)
- Device Access Project ID (from step 2c)

This saves your tokens to `nest_tokens.json` and lists your devices. Note the **device IDs** — you'll need them for `config.yaml`.

### Step 3: Get Your Location Coordinates

1. Find your home's latitude and longitude (use [Google Maps](https://maps.google.com), right-click your location)
2. The agent uses [Open-Meteo](https://open-meteo.com/) as its primary weather source — no API key needed
3. (Optional) For fallback weather, create a free [OpenWeatherMap](https://openweathermap.org/api) account and copy your API key

### Step 4: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g. "AI Thermostat")
4. Choose a username (must end in `bot`, e.g. `MyHomeThermostatBot`)
5. BotFather gives you a **bot token** — copy it
6. To get your **chat ID**:
   - Send any message to your new bot
   - Open `https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates` in a browser
   - Find `"chat":{"id":XXXXXXXX}` — that number is your chat ID
7. To add your family members:
   - Have them message the bot
   - Check `getUpdates` again for their chat IDs
   - Add all chat IDs to the whitelist in `config.yaml`

### Step 5: Configure

Edit `config.yaml` with your values:

```yaml
llm:
  server_exe: "C:\\path\\to\\llama-server.exe"         # Full path to llama-server executable
  model_path: "C:\\path\\to\\model.gguf"               # Full path to Qwen 4B GGUF model
  endpoint: "http://localhost:8080/v1/chat/completions"
  model: "gemma-4-e2b"
  temperature: 0.3
  top_p: 0.9
  max_tokens: 500
  timeout_seconds: 120

nest:
  tokens_path: "nest_tokens.json"
  devices:
    - name: "Upstairs Bedroom"
      device_id: "enterprises/YOUR_PROJECT_ID/devices/YOUR_DEVICE_ID"
    - name: "Downstairs Kitchen"
      device_id: "enterprises/YOUR_PROJECT_ID/devices/YOUR_DEVICE_ID"

weather:
  api_key: "YOUR_OPENWEATHERMAP_API_KEY"
  latitude: 33.0198
  longitude: -96.6989
  cache_minutes: 20
  stale_hours: 6

telegram:
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
  whitelisted_chat_ids:
    - 123456789       # Your chat ID
    - 987654321       # Family member's chat ID

comfort:
  summer_range: [75, 80]   # Cooling season target range (F)
  winter_range: [68, 72]   # Heating season target range (F)
  user_request_hours: 2    # Honor user requests this long, then re-evaluate

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

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 7: Run

```bash
start.bat
```

Or directly:

```bash
python agent.py
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Show current temperature, humidity, mode, and last decision for all zones |
| `/history` | Show the last 2 decisions |
| `/export` | Download the full climate log as a CSV file |
| Any text message | Treated as a natural language instruction (e.g. "set upstairs to 78") |

## How It Works

Every 20 minutes (configurable), the agent:

1. Reads indoor conditions from each Nest thermostat
2. Fetches outdoor weather and forecast
3. Checks for any recent Telegram messages from users
4. Starts the local LLM server (llama-server)
5. Sends all context to the AI, which returns a JSON decision per zone
6. Validates the decision against safety guardrails
7. Executes temperature changes via the Nest API
8. Stops the LLM server to free GPU memory
9. Logs everything to SQLite
10. Sends you a Telegram message if any temperature was changed

When you send a message via Telegram, the agent immediately runs an extra evaluation cycle with your request as priority context.

## Safety Guardrails

These are hard-coded and cannot be overridden by the AI:

| Guardrail | Value |
|-----------|-------|
| Minimum temperature | 65F (60F in vacation mode) |
| Maximum temperature | 80F (85F in vacation mode) |
| Max changes per hour | 6 |
| Manual override backoff | 120 minutes |

- **User requests bypass rate limits** — if you ask for a change, it always goes through
- **Manual override detection** — if someone changes the thermostat physically, the agent backs off for 2 hours
- **LLM response validation** — malformed JSON or out-of-range temperatures are rejected

## File Structure

```
AIThermostat/
├── agent.py           # Main brain — evaluation loop, LLM calls, guardrails
├── telegram_bot.py    # Telegram bot — commands, message handling
├── nest_api.py        # Nest SDM API wrapper — read state, set temperature
├── weather.py         # Weather client — Open-Meteo primary, OWM fallback
├── database.py        # SQLite logging — climate, decisions, messages, errors
├── location.py        # OwnTracks presence detection, vacation mode logic
├── llm_server.py      # On-demand llama-server lifecycle manager
├── nest_setup.py      # One-time setup script for Nest API tokens
├── test_qwen_4b.py    # LLM reliability test (18 scenarios)
├── config.yaml        # All configuration
├── requirements.txt   # Python dependencies
├── start.bat          # Windows startup script
├── PRODUCT.md         # Product decisions, pivots, and learnings
├── ARCHITECTURE.md    # System architecture and component details
├── tests/             # 95 unit tests (pytest)
├── DesignDOC/         # Original design document
└── thermostat.db      # SQLite database (created on first run)
```

## Troubleshooting

**"LLM server failed to start"**
- Check that `llama-server.exe` path in `llm_server.py` is correct
- Ensure the GGUF model file exists at the configured path
- Check GPU memory — close other GPU-intensive apps

**"Telegram bot did not register in time"**
- Check your bot token in `config.yaml` is correct
- Ensure your machine has internet access
- Check the Telegram API isn't blocked on your network

**Bot doesn't respond to messages**
- Verify your chat ID is in the `whitelisted_chat_ids` list
- Send `/status` first — if that works, the bot is running

**"Access token expired" / Nest API 401 errors**
- The agent auto-refreshes tokens, but if `nest_tokens.json` is corrupted, re-run `python nest_setup.py`

**Temperature changes don't go through**
- Check the rate limit — run `/history` to see recent decisions
- Look for "BLOCKED" entries in the logs indicating guardrail intervention
- Verify the thermostat mode (COOL/HEAT) matches what you're trying to do

## Privacy

All AI processing runs locally on your hardware. No data is sent to cloud AI services. The only external API calls are:
- **Nest SDM API** — to read/control your thermostat (Google)
- **Open-Meteo API** — to get weather data (no API key, no account needed)
- **Telegram Bot API** — to send/receive messages
