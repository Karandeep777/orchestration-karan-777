# Support Ticket Classifier and Router Agent

Automatically classifies, prioritizes, and routes support tickets using Azure GPT-4.1. Sends acknowledgements, logs all decisions, and ensures compliance with audit and privacy requirements.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:
- **Windows:**
  ```
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```
  source .venv/bin/activate
  ```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

- **Direct execution:**
  ```
  python code/agent.py
  ```
- **As a FastAPI server:**
  ```
  uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
  ```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`, `AGENT_ID`, `PROJECT_NAME`, `PROJECT_ID`, `SERVICE_NAME`, `SERVICE_VERSION`

**General**
- `ENVIRONMENT`

**Azure Key Vault**
- `USE_KEY_VAULT`, `KEY_VAULT_URI`, `AZURE_USE_DEFAULT_CREDENTIAL`

**Azure Authentication**
- `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`
- `LLM_MODELS` (JSON list for cost calculation)

**API Keys / Secrets**
- `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`

**Service Endpoints**
- `AZURE_CONTENT_SAFETY_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `AZURE_SEARCH_INDEX_NAME`

**Observability Database**
- `OBS_DATABASE_TYPE`, `OBS_AZURE_SQL_SERVER`, `OBS_AZURE_SQL_DATABASE`, `OBS_AZURE_SQL_PORT`, `OBS_AZURE_SQL_USERNAME`, `OBS_AZURE_SQL_PASSWORD`, `OBS_AZURE_SQL_SCHEMA`, `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**Agent-Specific**
- `REQUIRE_CONSENT`, `VALIDATION_CONFIG_PATH`, `VERSION`
- `CONTENT_SAFETY_ENABLED`, `CONTENT_SAFETY_SEVERITY_THRESHOLD`

See `.env.example` for descriptions and required/optional status.

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

---

### **POST** `/process_ticket`
- **Description:** Process a support ticket: classify, prioritize, route, acknowledge, and audit.
- **Request body:**
  ```
  {
    "ticket_id": "string (optional)",
    "channel": "string (optional)",
    "customer_id": "string (optional)",
    "customer_email": "string (optional)",
    "customer_tier": "string (optional)",
    "sla_status": "string (optional)",
    "ticket_subject": "string (optional)",
    "ticket_content": "string (required)",
    "product": "string (optional)",
    "consent": true|false (optional),
    "received_at": "string (optional)"
  }
  ```
- **Response:**
  ```
  {
    "category": "string|null",
    "priority": "string|null",
    "product_area": "string|null",
    "sentiment_flag": true|false|null,
    "duplicate_link": "string|null",
    "route_to": "string|null",
    "acknowledgement_status": "string|null",
    "audit_log": ["string", ...] | null,
    "errors": ["string", ...] | null,
    "success": true|false,
    "fallback_used": true|false|null
  }
  ```

- **Validation error (422) response:**
  ```
  {
    "success": false,
    "error": {
      "type": "ValidationError",
      "message": "Malformed JSON or validation error. ...",
      "tips": "Check input format, required fields, and try again."
    }
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t support-ticket-classifier-and-router-agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name support-ticket-classifier-and-router-agent support-ticket-classifier-and-router-agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs support-ticket-classifier-and-router-agent
```

### 7. Stop the container:
```
docker stop support-ticket-classifier-and-router-agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Support Ticket Classifier and Router Agent** — Instantly triage, classify, and route support tickets with audit-grade compliance and LLM-powered accuracy.
