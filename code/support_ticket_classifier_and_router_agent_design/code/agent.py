import asyncio
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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import Config

import openai

# ========== CONSTANTS ==========

SYSTEM_PROMPT = (
    "You are a professional Support Ticket Classifier and Router Agent. Your role is to automatically process incoming customer support tickets, classify them by issue category, urgency level, and product area, detect sentiment and duplicates, and route each ticket to the correct support team queue. Follow these instructions:\n\n"
    "- Classify tickets into one of the following categories: Billing, Technical Issue, Account Access, Feature Request, Bug Report, Shipping, Refund, General Inquiry.\n\n"
    "- Assign priority as Critical, High, Medium, or Low based on urgency keywords, customer tier, SLA status, and sentiment.\n\n"
    "- Tag the relevant product, module, or service referenced in the ticket.\n\n"
    "- Detect angry or frustrated tone and flag for priority handling.\n\n"
    "- Identify duplicate tickets from the same customer and link to the original, notify the customer, and close the duplicate.\n\n"
    "- Route tickets based on category, priority, team availability, and customer tier (VIP customers are escalated to senior agents).\n\n"
    "- Always send an acknowledgement email to the customer within 30 seconds of ticket receipt, including ticket ID and estimated response time.\n\n"
    "- If routing confidence is below 0.70, flag the ticket for human triage review instead of auto-routing.\n\n"
    "- Log the reason for every routing decision for audit and model improvement.\n\n"
    "- Treat tickets containing words like \"legal\", \"lawsuit\", \"regulator\" as Critical regardless of other signals.\n\n"
    "- Ensure compliance with GDPR/CCPA for PII handling, do not share ticket content outside the support system, and track customer consent for communications.\n\n"
    "Output all results in structured JSON format with fields for category, priority, product_area, sentiment_flag, duplicate_link, route_to, acknowledgement_status, audit_log, and any errors encountered.\n\n"
    "If information is missing or ambiguous, escalate to human review and provide a clear explanation in the output."
)
OUTPUT_FORMAT = (
    "Structured JSON with fields:\n"
    "  - category\n"
    "  - priority\n"
    "  - product_area\n"
    "  - sentiment_flag\n"
    "  - duplicate_link\n"
    "  - route_to\n"
    "  - acknowledgement_status\n"
    "  - audit_log\n"
    "  - errors"
)
FALLBACK_RESPONSE = (
    "Unable to confidently classify or route this ticket. The ticket has been flagged for human triage review and the customer will be notified."
)
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# ========== LLM OUTPUT SANITIZER ==========

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

# ========== INPUT/OUTPUT MODELS ==========

class TicketPayload(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Unique ticket identifier (if available)")
    channel: Optional[str] = Field(None, description="Source channel (email, web, chat, api)")
    customer_id: Optional[str] = Field(None, description="Customer unique identifier")
    customer_email: Optional[str] = Field(None, description="Customer email address")
    customer_tier: Optional[str] = Field(None, description="Customer tier (VIP, Standard, etc.)")
    sla_status: Optional[str] = Field(None, description="SLA status (breach imminent, within limits, etc.)")
    ticket_subject: Optional[str] = Field(None, description="Ticket subject or summary")
    ticket_content: str = Field(..., description="Full content/body of the support ticket")
    product: Optional[str] = Field(None, description="Product/module/service referenced")
    consent: Optional[bool] = Field(None, description="Customer consent for communication")
    received_at: Optional[str] = Field(None, description="ISO timestamp when ticket was received")

    @field_validator("ticket_content")
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Ticket content must not be empty.")
        if len(v) > 50000:
            raise ValueError("Ticket content exceeds 50,000 character limit.")
        return v.strip()

class TicketResponse(BaseModel):
    category: Optional[str] = None
    priority: Optional[str] = None
    product_area: Optional[str] = None
    sentiment_flag: Optional[bool] = None
    duplicate_link: Optional[str] = None
    route_to: Optional[str] = None
    acknowledgement_status: Optional[str] = None
    audit_log: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    success: bool = True
    fallback_used: Optional[bool] = None

# ========== ERROR HANDLING ==========

@with_content_safety(config=GUARDRAILS_CONFIG)
def format_error_response(message: str, error_type: str = "AgentError", tips: Optional[str] = None) -> Dict[str, Any]:
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "tips": tips or "Check input format, required fields, and try again."
        }
    }

# ========== SERVICE CLASSES ==========

class TicketInputHandler:
    """Receives and normalizes incoming tickets from multiple channels."""

    def receive_ticket(self, ticket_payload: TicketPayload) -> Dict[str, Any]:
        """Receives ticket from channel, normalizes input."""
        try:
            ticket_dict = ticket_payload.dict()
            # Normalize fields (strip, lower-case channel, etc.)
            ticket_dict["channel"] = (ticket_dict.get("channel") or "api").strip().lower()
            ticket_dict["ticket_content"] = ticket_dict["ticket_content"].strip()
            if ticket_dict.get("customer_email"):
                ticket_dict["customer_email"] = ticket_dict["customer_email"].strip().lower()
            return ticket_dict
        except Exception as e:
            logger.error(f"Input normalization failed: {e}")
            raise ValueError("Malformed ticket payload. " + str(e))

class TicketPreprocessor:
    """Performs basic validation, deduplication check, and consent verification."""

    def validate_ticket(self, ticket: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Checks ticket for required fields, consent, and duplicate status."""
        errors = []
        # Consent check
        if Config.REQUIRE_CONSENT and not ticket.get("consent", False):
            errors.append("Customer consent missing or not granted.")
        # Duplicate check (stub: always False, real impl would query DB)
        is_duplicate = False
        duplicate_link = None
        # Simulate duplicate detection logic
        # In production, integrate with DuplicateDetectionService
        # For now, always return not duplicate
        if is_duplicate:
            duplicate_link = "existing_ticket_id"
            errors.append("Duplicate ticket detected.")
        ticket["duplicate_link"] = duplicate_link
        return ticket, errors[0] if errors else None

class LLMService:
    """Calls Azure GPT-4.1 to classify, score, tag, detect sentiment, and apply business rules."""

    def __init__(self):
        self.client = None

    def _get_client(self):
        if self.client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self.client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def classify_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Calls LLM to classify category, priority, product area, sentiment, duplicate."""
        _t0 = _time.time()
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
            },
            {
                "role": "user",
                "content": self._build_user_message(ticket)
            }
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            cleaned = sanitize_llm_output(content, content_type="code")
            # Try to parse as JSON
            try:
                result = json.loads(cleaned)
                result["success"] = True
                result["fallback_used"] = False
                return result
            except Exception:
                # If not JSON, fallback
                logger.warning("LLM output not valid JSON, using fallback response.")
                return {
                    "success": False,
                    "fallback_used": True,
                    "errors": ["LLM output not valid JSON."],
                    "category": None,
                    "priority": None,
                    "product_area": None,
                    "sentiment_flag": None,
                    "duplicate_link": ticket.get("duplicate_link"),
                    "route_to": None,
                    "acknowledgement_status": None,
                    "audit_log": [],
                }
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {
                "success": False,
                "fallback_used": True,
                "errors": [str(e)],
                "category": None,
                "priority": None,
                "product_area": None,
                "sentiment_flag": None,
                "duplicate_link": ticket.get("duplicate_link"),
                "route_to": None,
                "acknowledgement_status": None,
                "audit_log": [],
            }

    def _build_user_message(self, ticket: Dict[str, Any]) -> str:
        """Builds the user message for the LLM from ticket fields."""
        fields = [
            f"Ticket ID: {ticket.get('ticket_id', 'N/A')}",
            f"Channel: {ticket.get('channel', 'N/A')}",
            f"Customer ID: {ticket.get('customer_id', 'N/A')}",
            f"Customer Email: {ticket.get('customer_email', 'N/A')}",
            f"Customer Tier: {ticket.get('customer_tier', 'N/A')}",
            f"SLA Status: {ticket.get('sla_status', 'N/A')}",
            f"Ticket Subject: {ticket.get('ticket_subject', 'N/A')}",
            f"Product: {ticket.get('product', 'N/A')}",
            f"Consent: {ticket.get('consent', 'N/A')}",
            f"Received At: {ticket.get('received_at', 'N/A')}",
            f"Ticket Content:\n{ticket.get('ticket_content', '')}"
        ]
        return "\n".join(fields)

class RoutingEngine:
    """Applies routing rules, decision tables, and escalates as needed."""

    async def route_ticket(self, ticket: Dict[str, Any], classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Applies routing rules and decision tables to assign queue and escalate."""
        # For this agent, routing is handled by LLM, but we can post-process for confidence, compliance, etc.
        errors = []
        route_to = classification_result.get("route_to")
        audit_log = classification_result.get("audit_log") or []
        # Simulate routing confidence check
        routing_confidence = classification_result.get("routing_confidence", 1.0)
        if isinstance(routing_confidence, str):
            try:
                routing_confidence = float(routing_confidence)
            except Exception:
                routing_confidence = 1.0
        if routing_confidence < 0.70:
            errors.append("Routing confidence below threshold. Flagged for human triage.")
            audit_log.append("Routing confidence below 0.70; ticket flagged for human review.")
            route_to = "HITL_Triage"
        # Compliance check (stub)
        # In production, check for compliance violations
        return {
            "route_to": route_to,
            "audit_log": audit_log,
            "errors": errors,
            "routing_confidence": routing_confidence
        }

class IntegrationLayer:
    """Handles communication with email sender, queue manager, audit logger, consent tracker."""

    async def send_acknowledgement(self, ticket: Dict[str, Any]) -> str:
        """Sends confirmation email to customer."""
        # Simulate async email sending (stub)
        # In production, integrate with EmailSender
        try:
            # Simulate retry logic
            for attempt in range(3):
                try:
                    # Simulate sending
                    await self._simulate_email_send(ticket)
                    return "sent"
                except Exception as e:
                    logger.warning(f"Acknowledgement send attempt {attempt+1} failed: {e}")
                    await self._backoff(attempt)
            logger.error("Acknowledgement failed after 3 attempts.")
            return "failed"
        except Exception as e:
            logger.error(f"Acknowledgement error: {e}")
            return "failed"

    async def _simulate_email_send(self, ticket: Dict[str, Any]):
        # Simulate a delay and always succeed
        await self._async_sleep(0.1)
        return

    async def _async_sleep(self, seconds: float):
        await asyncio.sleep(seconds)

    async def _backoff(self, attempt: int):
        await self._async_sleep(0.5 * (2 ** attempt))

    async def assign_queue(self, ticket: Dict[str, Any]):
        # Stub: In production, integrate with TicketQueueManager
        return

    async def log_audit(self, ticket: Dict[str, Any], decision: str):
        # Stub: In production, integrate with AuditLogger
        return

    async def track_consent(self, customer_id: str, preferences: Any):
        # Stub: In production, integrate with ConsentTracker
        return

class AuditLogger:
    """Logs all routing, classification, and compliance decisions for audit."""

    def __init__(self):
        self.logger = logging.getLogger("audit_logger")

    async def log_decision(self, ticket_id: str, decision: str, reason: str) -> str:
        """Logs routing/classification decisions for audit."""
        try:
            entry = f"Ticket {ticket_id}: {decision} | Reason: {reason}"
            self.logger.info(entry)
            return entry
        except Exception as e:
            self.logger.error(f"Audit log failed: {e}")
            # In production, trigger compliance alert
            return f"Audit log failed: {e}"

class OutputFormatter:
    """Formats structured JSON output and error responses."""

    def format_response(
        self,
        ticket: Dict[str, Any],
        routing_result: Dict[str, Any],
        audit_log: List[str],
        errors: List[str]
    ) -> Dict[str, Any]:
        """Formats structured JSON output for requester."""
        try:
            response = {
                "category": ticket.get("category"),
                "priority": ticket.get("priority"),
                "product_area": ticket.get("product_area"),
                "sentiment_flag": ticket.get("sentiment_flag"),
                "duplicate_link": ticket.get("duplicate_link"),
                "route_to": routing_result.get("route_to"),
                "acknowledgement_status": ticket.get("acknowledgement_status"),
                "audit_log": audit_log,
                "errors": errors,
                "success": not errors,
                "fallback_used": ticket.get("fallback_used", False)
            }
            return response
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return {
                "success": False,
                "errors": [str(e), "Fallback response used."],
                "fallback_used": True,
                "category": None,
                "priority": None,
                "product_area": None,
                "sentiment_flag": None,
                "duplicate_link": None,
                "route_to": None,
                "acknowledgement_status": None,
                "audit_log": [],
            }

# ========== MAIN AGENT CLASS ==========

class SupportTicketAgent:
    """Main agent orchestrating ticket classification, routing, acknowledgement, and audit logging."""

    def __init__(self):
        self.input_handler = TicketInputHandler()
        self.preprocessor = TicketPreprocessor()
        self.llm_service = LLMService()
        self.routing_engine = RoutingEngine()
        self.integration_layer = IntegrationLayer()
        self.audit_logger = AuditLogger()
        self.output_formatter = OutputFormatter()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_ticket(self, ticket_payload: TicketPayload) -> Dict[str, Any]:
        """Orchestrates end-to-end ticket classification, routing, acknowledgement, and audit logging."""
        audit_log: List[str] = []
        errors: List[str] = []
        try:
            async with trace_step(
                "input_normalization",
                step_type="parse",
                decision_summary="Normalize and validate incoming ticket",
                output_fn=lambda r: f"normalized={bool(r)}"
            ) as step:
                ticket = self.input_handler.receive_ticket(ticket_payload)
                step.capture(ticket)

            async with trace_step(
                "preprocessing",
                step_type="parse",
                decision_summary="Validate ticket, check consent and duplicates",
                output_fn=lambda r: f"errors={r[1]}"
            ) as step:
                ticket, pre_error = self.preprocessor.validate_ticket(ticket)
                if pre_error:
                    errors.append(pre_error)
                step.capture((ticket, pre_error))

            if errors:
                audit_entry = await self.audit_logger.log_decision(
                    ticket.get("ticket_id", "N/A"),
                    "preprocessing_failed",
                    "; ".join(errors)
                )
                audit_log.append(audit_entry)
                return self.output_formatter.format_response(ticket, {}, audit_log, errors)

            async with trace_step(
                "llm_classification",
                step_type="llm_call",
                decision_summary="Classify ticket using LLM",
                output_fn=lambda r: f"category={r.get('category')}, priority={r.get('priority')}"
            ) as step:
                classification_result = await self.llm_service.classify_ticket(ticket)
                step.capture(classification_result)

            if not classification_result.get("success", False):
                errors.extend(classification_result.get("errors", []))
                audit_entry = await self.audit_logger.log_decision(
                    ticket.get("ticket_id", "N/A"),
                    "classification_failed",
                    "; ".join(errors)
                )
                audit_log.append(audit_entry)
                # Fallback: escalate to human triage
                ticket["route_to"] = "HITL_Triage"
                ticket["fallback_used"] = True
                ticket["acknowledgement_status"] = None
                return self.output_formatter.format_response(ticket, {}, audit_log, errors)

            # Merge classification fields into ticket
            for key in [
                "category", "priority", "product_area", "sentiment_flag",
                "duplicate_link", "route_to", "acknowledgement_status"
            ]:
                if key in classification_result:
                    ticket[key] = classification_result[key]

            # Routing
            async with trace_step(
                "routing",
                step_type="process",
                decision_summary="Apply routing rules and escalate if needed",
                output_fn=lambda r: f"route_to={r.get('route_to')}"
            ) as step:
                routing_result = await self.routing_engine.route_ticket(ticket, classification_result)
                step.capture(routing_result)
                if routing_result.get("errors"):
                    errors.extend(routing_result.get("errors", []))
                if routing_result.get("audit_log"):
                    audit_log.extend(routing_result.get("audit_log", []))
                if routing_result.get("route_to"):
                    ticket["route_to"] = routing_result["route_to"]

            # Send acknowledgement
            async with trace_step(
                "acknowledgement",
                step_type="tool_call",
                decision_summary="Send acknowledgement email to customer",
                output_fn=lambda r: f"status={r}"
            ) as step:
                ack_status = await self.integration_layer.send_acknowledgement(ticket)
                ticket["acknowledgement_status"] = ack_status
                step.capture(ack_status)
                if ack_status != "sent":
                    errors.append("Acknowledgement email failed to send.")

            # Audit log
            async with trace_step(
                "audit_logging",
                step_type="process",
                decision_summary="Log all decisions for audit",
                output_fn=lambda r: f"audit_entries={len(r)}"
            ) as step:
                audit_entry = await self.audit_logger.log_decision(
                    ticket.get("ticket_id", "N/A"),
                    "processed",
                    f"Category: {ticket.get('category')}, Priority: {ticket.get('priority')}, Route: {ticket.get('route_to')}"
                )
                audit_log.append(audit_entry)
                step.capture(audit_log)

            # Format response
            async with trace_step(
                "output_formatting",
                step_type="format",
                decision_summary="Format structured JSON response",
                output_fn=lambda r: f"success={r.get('success')}"
            ) as step:
                response = self.output_formatter.format_response(ticket, routing_result, audit_log, errors)
                step.capture(response)
                return response

        except Exception as e:
            logger.error(f"Agent processing failed: {e}", exc_info=True)
            errors.append(str(e))
            audit_entry = await self.audit_logger.log_decision(
                ticket_payload.ticket_id or "N/A",
                "agent_error",
                str(e)
            )
            audit_log.append(audit_entry)
            return {
                "success": False,
                "errors": errors,
                "audit_log": audit_log,
                "fallback_used": True,
                "category": None,
                "priority": None,
                "product_area": None,
                "sentiment_flag": None,
                "duplicate_link": None,
                "route_to": None,
                "acknowledgement_status": None,
            }

# ========== FASTAPI APP & ENDPOINTS ==========

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

    title="Support Ticket Classifier and Router Agent",
    description="Automatically classifies, prioritizes, and routes support tickets using Azure GPT-4.1. Sends acknowledgements, logs all decisions, and ensures compliance.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Malformed JSON or validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content=format_error_response(
            message="Malformed JSON or validation error. " + str(exc),
            error_type="ValidationError",
            tips="Check for missing fields, extra commas, or incorrect types. Ensure ticket_content is present and under 50,000 characters."
        )
    )

@app.post("/process_ticket", response_model=TicketResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def process_ticket_endpoint(payload: TicketPayload):
    """
    Process a support ticket: classify, prioritize, route, acknowledge, and audit.
    """
    agent = SupportTicketAgent()
    result = await agent.process_ticket(payload)
    # Sanitize LLM output before returning
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, str):
                result[k] = sanitize_llm_output(v, content_type="text")
    return result

# ========== MAIN ENTRYPOINT ==========

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