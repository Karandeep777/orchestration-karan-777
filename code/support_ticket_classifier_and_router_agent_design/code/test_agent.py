
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent
from agent import SupportTicketAgent, TicketPayload, TicketInputHandler, TicketPreprocessor, LLMService, RoutingEngine, Config, app

from fastapi.testclient import TestClient

@pytest.fixture
def valid_ticket_payload():
    return TicketPayload(
        ticket_id="T123",
        channel="email",
        customer_id="C456",
        customer_email="user@example.com",
        customer_tier="VIP",
        sla_status="within limits",
        ticket_subject="Cannot access account",
        ticket_content="I cannot access my account and need urgent help.",
        product="ProductX",
        consent=True,
        received_at="2024-05-01T12:00:00Z"
    )

@pytest.fixture
def valid_ticket_dict(valid_ticket_payload):
    return valid_ticket_payload.dict()

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_process_valid_ticket_end_to_end(valid_ticket_payload):
    """Validates the full workflow of process_ticket with a well-formed, consented, non-duplicate ticket."""
    agent_instance = SupportTicketAgent()
    # Patch LLMService.classify_ticket to return a valid classification result
    mock_llm_result = {
        "category": "Account Access",
        "priority": "High",
        "product_area": "ProductX",
        "sentiment_flag": True,
        "duplicate_link": None,
        "route_to": "Account_Support",
        "acknowledgement_status": "sent",
        "audit_log": ["LLM classified ticket"],
        "errors": [],
        "success": True,
        "fallback_used": False
    }
    with patch.object(agent_instance.llm_service, "classify_ticket", AsyncMock(return_value=mock_llm_result)), \
         patch.object(agent_instance.integration_layer, "send_acknowledgement", AsyncMock(return_value="sent")), \
         patch.object(agent_instance.audit_logger, "log_decision", AsyncMock(return_value="audit entry")):
        result = await agent_instance.process_ticket(valid_ticket_payload)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["category"] is not None
    assert result["priority"] is not None
    assert result["route_to"] is not None
    assert result["acknowledgement_status"] == "sent"
    assert isinstance(result["audit_log"], list) and len(result["audit_log"]) > 0
    assert not result.get("errors")
    assert result["fallback_used"] is False

@pytest.mark.asyncio
async def test_reject_ticket_with_missing_consent_when_required(valid_ticket_payload, monkeypatch):
    """Ensures process_ticket returns an error if consent is required but not provided in the ticket."""
    agent_instance = SupportTicketAgent()
    # Set Config.REQUIRE_CONSENT = True
    monkeypatch.setattr(agent.Config, "REQUIRE_CONSENT", True)
    payload = valid_ticket_payload.copy(update={"consent": None})
    # AUTO-FIXED invalid syntax: with patch.object(agent_instance.audit_logger, "log_decision", AsyncMock(return_value="audit entry"):
    result = await agent_instance.process_ticket(payload)
    assert result["success"] is False
    assert "Customer consent missing or not granted." in result["errors"][0]
    assert result["fallback_used"] is False
    assert isinstance(result["audit_log"], list) and len(result["audit_log"]) > 0

@pytest.mark.asyncio
async def test_handle_llm_output_not_valid_json(valid_ticket_payload):
    """Ensures that if the LLM returns output that cannot be parsed as JSON, the agent falls back to HITL triage."""
    agent_instance = SupportTicketAgent()
    # Patch LLMService.classify_ticket to return fallback (simulate non-JSON output)
    fallback_result = {
        "success": False,
        "fallback_used": True,
        "errors": ["LLM output not valid JSON."],
        "category": None,
        "priority": None,
        "product_area": None,
        "sentiment_flag": None,
        "duplicate_link": None,
        "route_to": None,
        "acknowledgement_status": None,
        "audit_log": [],
    }
    with patch.object(agent_instance.llm_service, "classify_ticket", AsyncMock(return_value=fallback_result)), \
         patch.object(agent_instance.audit_logger, "log_decision", AsyncMock(return_value="audit entry")):
        result = await agent_instance.process_ticket(valid_ticket_payload)
    assert result["success"] is False
    assert result["fallback_used"] is True
    assert result["route_to"] == "HITL_Triage"
    assert "LLM output not valid JSON." in result["errors"][0]

def test_health_check_endpoint_returns_ok(test_client):
    """Verifies the /health endpoint returns a 200 status and correct payload."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

def test_process_ticket_endpoint_returns_200_and_expected_fields(test_client, valid_ticket_payload):
    """Verifies the /process_ticket endpoint returns a 200 status and all expected fields for a valid ticket."""
    # Patch SupportTicketAgent.process_ticket to return a valid response
    expected_response = {
        "category": "Billing",
        "priority": "High",
        "product_area": "ProductX",
        "sentiment_flag": False,
        "duplicate_link": None,
        "route_to": "Billing_Support",
        "acknowledgement_status": "sent",
        "audit_log": ["audit entry"],
        "errors": [],
        "success": True,
        "fallback_used": False
    }
    # AUTO-FIXED invalid syntax: with patch("agent.SupportTicketAgent.process_ticket", AsyncMock(return_value=expected_response):
    resp = test_client.post("/process_ticket", data=valid_ticket_payload.model_dump_json(), headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    for field in [
        "category", "priority", "product_area", "sentiment_flag", "duplicate_link",
        "route_to", "acknowledgement_status", "audit_log", "errors", "success", "fallback_used"
    ]:
        assert field in data

def test_validation_error_for_missing_required_ticket_content(test_client):
    """Ensures the endpoint returns a 422 error and proper error message if ticket_content is missing."""
    payload = {
        "ticket_id": "T123",
        "channel": "email",
        "customer_id": "C456",
        "customer_email": "user@example.com",
        "customer_tier": "VIP",
        "sla_status": "within limits",
        "ticket_subject": "Cannot access account",
        # "ticket_content" is missing
        "product": "ProductX",
        "consent": True,
        "received_at": "2024-05-01T12:00:00Z"
    }
    resp = test_client.post("/process_ticket", data=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    data = resp.json()
    assert "ticket_content" in data["error"]["message"]

def test_ticket_input_handler_normalizes_input_fields():
    """Unit test for TicketInputHandler.receive_ticket to ensure normalization of channel and customer_email."""
    handler = TicketInputHandler()
    payload = TicketPayload(
        ticket_id="T1",
        channel="EMAIL ",
        customer_id="C1",
        customer_email="USER@EXAMPLE.COM ",
        customer_tier="Standard",
        sla_status="within limits",
        ticket_subject="Test",
        ticket_content="Test content",
        product="ProductY",
        consent=True,
        received_at="2024-05-01T12:00:00Z"
    )
    result = handler.receive_ticket(payload)
    assert result["channel"] == "email"
    assert result["customer_email"] == "user@example.com"

def test_ticket_preprocessor_detects_missing_consent(monkeypatch):
    """Unit test for TicketPreprocessor.validate_ticket to ensure it returns an error if consent is missing and required."""
    preprocessor = TicketPreprocessor()
    monkeypatch.setattr(agent.Config, "REQUIRE_CONSENT", True)
    ticket = {
        "ticket_id": "T2",
        "ticket_content": "Help me",
        "consent": None
    }
    result, error = preprocessor.validate_ticket(ticket)
    assert error == "Customer consent missing or not granted."

@pytest.mark.asyncio
async def test_llmservice_classify_ticket_returns_fallback_on_api_key_missing(monkeypatch):
    """Unit test for LLMService.classify_ticket to ensure it returns fallback response if AZURE_OPENAI_API_KEY is missing."""
    llm = LLMService()
    monkeypatch.setattr(agent.Config, "AZURE_OPENAI_API_KEY", "")
    ticket = {
        "ticket_id": "T3",
        "ticket_content": "Test ticket"
    }
    result = await llm.classify_ticket(ticket)
    assert result["success"] is False
    assert result["fallback_used"] is True
    assert "AZURE_OPENAI_API_KEY not configured" in result["errors"][0]

@pytest.mark.asyncio
async def test_routingengine_flags_low_routing_confidence():
    """Unit test for RoutingEngine.route_ticket to ensure tickets with routing_confidence < 0.70 are flagged for human triage."""
    engine = RoutingEngine()
    ticket = {
        "ticket_id": "T4",
        "ticket_content": "Test ticket"
    }
    classification_result = {
        "route_to": "SomeTeam",
        "audit_log": [],
        "routing_confidence": 0.5
    }
    result = await engine.route_ticket(ticket, classification_result)
    assert result["route_to"] == "HITL_Triage"
    assert any("Routing confidence below threshold." in err for err in result["errors"])