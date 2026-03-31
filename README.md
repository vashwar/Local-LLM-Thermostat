# AI Thermostat Agent

An autonomous smart climate controller that uses a locally-hosted AI (Qwen 4B via llama.cpp) to manage Nest thermostats. It reads indoor conditions, outdoor weather forecasts, and natural language instructions from a Telegram bot to make intelligent temperature decisions — all running 100% locally with zero cloud AI costs.

```
start.bat
  └─> python agent.py
        └─> asyncio.gather(
              agent_loop(),              # 20-min scheduled evaluation
              telegram_bot.start_bot()   # Telegram polling
            )

Single Evaluation Cycle:
  1. Read indoor temp/humidity from Nest API (per zone)
  2. Read outdoor weather + forecast from OpenWeatherMap
  3. Read recent Telegram messages for user requests
  4. Start llama-server (on-demand, GPU)
  5. Send context to LLM → get JSON decision
  6. Validate response + check guardrails
  7. Execute temperature change via Nest API (if needed)
  8. Stop llama-server (free GPU memory)
  9. Log everything to SQLite
```

## Features

- **Multi-zone support** — manages multiple Nest thermostats independently (e.g. Upstairs Bedroom, Downstairs Kitchen)
- **Local AI brain** — Qwen 4B runs on your GPU via llama.cpp, no cloud API needed
- **On-demand LLM** — llama-server starts only during evaluations, then shuts down to free GPU memory
- **Telegram bot** — send natural language commands ("set upstairs to 78"), get status, export data
- **User requests are priority** — the AI always follows your explicit instructions over its own logic
- **Safety guardrails** — hard-coded temp bounds (60-85F), rate limiting, manual override detection
- **Weather-aware** — considers outdoor temperature, forecast, and weather alerts
- **Schedule-aware** — knows sleep/wake times, work hours, weekends
- **Climate logging** — full SQLite history for analysis and weekly reports

## Prerequisites

- **Windows 10/11** (tested on Windows 11)
- **Python 3.10+**
- **NVIDIA GPU** with enough VRAM for Qwen 4B (~3GB)
- **llama.cpp** — `llama-server.exe` compiled or downloaded
- **A Qwen 4B GGUF model** — e.g. `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
- **Google Nest thermostat(s)** with Smart Device Management API access
- **OpenWeatherMap** free account
- **Telegram** account

## Setup Guide

### Step 1: Install llama.cpp

1. Download a prebuilt release from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) (pick the CUDA version matching your GPU)
2. Extract `llama-server.exe` to a folder, e.g. `C:\VashwarTests\EmailOrganizer\llama-server\`
3. Download the Qwen 4B GGUF model from [HuggingFace](https://huggingface.co/Qwen) and place it in a `models\` folder, e.g. `C:\VashwarTests\EmailOrganizer\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
4. Update the paths in `llm_server.py` if your locations differ:
   ```python
   _server_exe = r"C:\path\to\llama-server.exe"
   _model_path = r"C:\path\to\model.gguf"
   ```

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

### Step 3: Get an OpenWeatherMap API Key

1. Create a free account at [OpenWeatherMap](https://openweathermap.org/api)
2. Go to **API Keys** in your account settings
3. Copy your API key (free tier allows 1,000 calls/day — this agent uses ~72/day)
4. Find your home's latitude and longitude (use [Google Maps](https://maps.google.com), right-click your location)

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
  endpoint: "http://localhost:8080/v1/chat/completions"
  model: "qwen4b"
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
| Minimum temperature | 65F |
| Maximum temperature | 80F |
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
├── weather.py         # OpenWeatherMap client with caching
├── database.py        # SQLite logging — climate, decisions, messages, errors
├── llm_server.py      # On-demand llama-server lifecycle manager
├── nest_setup.py      # One-time setup script for Nest API tokens
├── test_qwen_4b.py    # LLM reliability test (18 scenarios)
├── config.yaml        # All configuration
├── nest_tokens.json   # Nest API credentials (generated by nest_setup.py)
├── requirements.txt   # Python dependencies
├── start.bat          # Windows startup script
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
- **OpenWeatherMap API** — to get weather data
- **Telegram Bot API** — to send/receive messages
