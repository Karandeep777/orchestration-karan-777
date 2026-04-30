import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import Config

# Constants from user prompt template
SYSTEM_PROMPT = (
    "You are a professional Ticket Escalation and Follow-up Agent for customer service operations. Your role is to monitor all open support tickets in real time, proactively communicate with customers when SLAs are at risk, escalate overdue tickets to senior agents or managers, and ensure resolved tickets are closed with a satisfaction survey. \n\n"
    "Task Instructions:\n\n"
    "- Continuously track every open ticket against its SLA deadline by priority tier.\n"
    "- Send proactive, clearly identified follow-up messages to customers when response is delayed, ensuring no more than 2 automated follow-ups per ticket.\n"
    "- Escalate tickets to senior agents or managers when SLA breach is imminent (within 30 minutes) or has occurred, including all required ticket and customer details in the alert.\n"
    "- Detect inactivity on tickets (no agent activity for a configurable threshold) and alert the team lead.\n"
    "- After a ticket is marked resolved, send a CSAT survey (1-5 rating and optional comment) to the customer 1 hour later, ensuring the survey is optional and does not block closure.\n"
    "- Maintain an audit trail for every escalation event, including reason, notified parties, and timestamp.\n"
    "- Push daily SLA compliance and escalation summary reports to manager email or Slack.\n"
    "- Always include the ticket ID and original issue summary in every communication.\n"
    "- If a customer responds to a follow-up with new information, re-open the ticket and notify the assigned agent immediately.\n"
    "- Respect customer opt-out preferences for follow-up emails.\n"
    "- Ensure all communications are GDPR-compliant and clearly identified as system-generated.\n\n"
    "Output Format:\n"
    "- Use clear, concise, and professional language.\n"
    "- For customer communications, always include ticket ID, issue summary, and next steps.\n"
    "- For escalation alerts, include ticket ID, customer name/tier, issue summary, time open, SLA deadline, and assigned agent.\n"
    "- For CSAT surveys, provide a 1-5 rating scale and an optional comment field.\n"
    "- For reports, summarize SLA compliance, breaches, and escalation events.\n\n"
    "Fallback Response:\n"
    "- If required information is not available or an action cannot be completed, respond with a clear message indicating the limitation and escalate to a human operator if necessary."
)
OUTPUT_FORMAT = (
    "- Customer communications: text email/message with ticket ID, issue summary, and status update.\n"
    "- Escalation alerts: structured notification with all required ticket and customer details.\n"
    "- CSAT survey: email/message with rating scale (1-5) and optional comment.\n"
    "- Reports: summary table or bullet points of SLA compliance and escalation events."
)
FALLBACK_RESPONSE = (
    "\"I'm unable to complete this action due to missing information or a system limitation. A human support agent will follow up with you as soon as possible.\""
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# LLM utility: output sanitizer
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# Input/Output Models
class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[str] = Field(None, description="Agent response or output")
    error: Optional[str] = Field(None, description="Error message, if any")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input or retrying")

# No dynamic user input required for main agent operation (system prompt is fixed)
# If you want to expose dynamic endpoints for ticket actions, add request models here.

# Observability lifespan function
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Ticket Escalation and Follow-up Agent",
    description="Automates ticket monitoring, follow-up, escalation, inactivity detection, survey delivery, and reporting for customer service operations.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# JSON error handler for malformed requests
@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "tips": "Check for missing quotes, commas, or brackets. Ensure your request matches the expected schema.",
            "details": exc.errors(),
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Input validation failed.",
            "tips": "Check your input fields and types. Refer to the API docs for the correct schema.",
            "details": exc.errors(),
        },
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

class LLMService:
    """Handles LLM calls for communication template generation and responses."""
    def __init__(self):
        self._client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            import openai
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(self, context: Optional[str] = None) -> str:
        """
        Calls the LLM with the system prompt and optional context.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT}
        ]
        if context:
            messages.append({"role": "user", "content": context})
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        client = self.get_llm_client()
        response = await client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4o",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4o",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return sanitize_llm_output(content, content_type="text")

class TicketEscalationAgent:
    """
    Main agent class orchestrating ticket monitoring, follow-up, escalation, inactivity detection, survey delivery, and reporting.
    """
    def __init__(self):
        self.llm_service = LLMService()
        # Placeholders for orchestration, business logic, and integration clients
        # In a real implementation, these would be initialized with actual adapters/services
        # For this agent, LLM is used for communication generation only

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process(self) -> Dict[str, Any]:
        """
        Main agent entry-point: generates a professional communication template or report as per the system prompt.
        """
        async with trace_step(
            "llm_generate_response",
            step_type="llm_call",
            decision_summary="Generate communication template or report using LLM",
            output_fn=lambda r: f"result={str(r)[:100]}",
        ) as step:
            try:
                result = await self.llm_service.generate_response()
                step.capture(result)
                return {
                    "success": True,
                    "result": result,
                }
            except Exception as e:
                step.capture(str(e))
                return {
                    "success": False,
                    "error": str(e),
                    "tips": "Check LLM configuration, API keys, and input context. If the issue persists, contact support.",
                }

agent = TicketEscalationAgent()

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint():
    """
    Endpoint to trigger the agent's main process (no user input required) 
    """
    async with trace_step(
        "process_query_endpoint",
        step_type="process",
        decision_summary="Invoke agent process for ticket escalation and follow-up",
        output_fn=lambda r: f"success={r.get('success', False)}",
    ) as step:
        try:
            result = await agent.process()
            step.capture(result)
            if not result.get("success"):
                return QueryResponse(
                    success=False,
                    error=result.get("error") or FALLBACK_RESPONSE,
                    tips=result.get("tips") or "Try again or contact support.",
                )
            return QueryResponse(
                success=True,
                result=result.get("result"),
            )
        except Exception as e:
            step.capture(str(e))
            return QueryResponse(
                success=False,
                error=str(e),
                tips="An unexpected error occurred. Please try again or contact support.",
            )

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())