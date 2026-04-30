
import asyncio
import logging
import importlib

logger = logging.getLogger(__name__)

# Dynamically import agent modules
ticket_escalation_module = importlib.import_module(
    "code.ticket_escalation_and_follow-up_agent_design.agent"
)
support_ticket_classifier_module = importlib.import_module(
    "code.support_ticket_classifier_and_router_agent_design.agent"
)

# Get references to agent entrypoints
# Step 0: Ticket Escalation and Follow-up Agent
# Entrypoint: async def query_endpoint() -> dict
ticket_escalation_agent = getattr(ticket_escalation_module, "agent", None)
ticket_escalation_query_endpoint = getattr(
    ticket_escalation_module, "query_endpoint", None
)

# Step 1: Support Ticket Classifier and Router Agent
# Entrypoint: async def process_ticket_endpoint(payload: TicketPayload) -> dict
support_ticket_process_ticket_endpoint = getattr(
    support_ticket_classifier_module, "process_ticket_endpoint", None
)
TicketPayload = getattr(
    support_ticket_classifier_module, "TicketPayload", None
)

class OrchestrationEngine:
    """
    Orchestrates the workflow:
      1. Calls Ticket Escalation and Follow-up Agent (no input required).
      2. If successful, uses its output as the 'ticket_content' for the Support Ticket Classifier and Router Agent.
      3. Returns the final result, collecting errors from both steps.
    """

    def __init__(self):
        self.logger = logger

    async def execute(self, input_data: dict = None) -> dict:
        """
        Orchestration entrypoint.
        Args:
            input_data: dict (not used, as first agent requires no input)
        Returns:
            dict: Final output from the Support Ticket Classifier and Router Agent, plus error details if any.
        """
        errors = []
        step_results = {}

        # Step 1: Ticket Escalation and Follow-up Agent
        try:
            self.logger.info("Starting Ticket Escalation and Follow-up Agent...")
            # This agent expects no input and returns a dict with 'success', 'result', etc.
            escalation_result = await ticket_escalation_query_endpoint()
            step_results["ticket_escalation"] = escalation_result
            if not escalation_result.get("success", False):
                errors.append({
                    "step": "ticket_escalation",
                    "error": escalation_result.get("error"),
                    "tips": escalation_result.get("tips"),
                })
        except Exception as e:
            self.logger.exception("Error in Ticket Escalation and Follow-up Agent")
            escalation_result = {
                "success": False,
                "error": str(e),
                "tips": "Exception occurred in ticket escalation agent."
            }
            step_results["ticket_escalation"] = escalation_result
            errors.append({
                "step": "ticket_escalation",
                "error": str(e),
                "tips": "Exception occurred in ticket escalation agent."
            })

        # Step 2: Support Ticket Classifier and Router Agent
        classifier_result = None
        if escalation_result.get("success", False):
            # Use the 'result' field as ticket_content for the classifier agent
            ticket_content = escalation_result.get("result", "")
            if not ticket_content:
                errors.append({
                    "step": "support_ticket_classifier",
                    "error": "No result from escalation agent to use as ticket_content.",
                    "tips": "Check escalation agent output."
                })
            else:
                try:
                    self.logger.info("Starting Support Ticket Classifier and Router Agent...")
                    # Build TicketPayload with only required field 'ticket_content'
                    payload_kwargs = {"ticket_content": ticket_content}
                    # Optionally, add more fields if desired from input_data or escalation_result
                    payload = TicketPayload(**payload_kwargs)
                    classifier_result = await support_ticket_process_ticket_endpoint(payload)
                    step_results["support_ticket_classifier"] = classifier_result
                    if not classifier_result.get("success", False):
                        errors.append({
                            "step": "support_ticket_classifier",
                            "error": classifier_result.get("errors"),
                            "tips": "See audit_log for details." if "audit_log" in classifier_result else None
                        })
                except Exception as e:
                    self.logger.exception("Error in Support Ticket Classifier and Router Agent")
                    classifier_result = {
                        "success": False,
                        "errors": [str(e)],
                        "audit_log": [],
                        "fallback_used": True,
                        "category": None,
                        "priority": None,
                        "product_area": None,
                        "sentiment_flag": None,
                        "duplicate_link": None,
                        "route_to": None,
                        "acknowledgement_status": None,
                    }
                    step_results["support_ticket_classifier"] = classifier_result
                    errors.append({
                        "step": "support_ticket_classifier",
                        "error": str(e),
                        "tips": "Exception occurred in support ticket classifier agent."
                    })
        else:
            # Escalation agent failed, so we skip classifier agent
            classifier_result = None

        # Compose final result
        final_result = {
            "ticket_escalation_result": step_results.get("ticket_escalation"),
            "support_ticket_classifier_result": step_results.get("support_ticket_classifier"),
            "errors": errors if errors else None
        }
        # For convenience, if classifier_result exists, return its output at top level
        if classifier_result is not None:
            final_result.update(classifier_result)
        return final_result

# Convenience function for direct use
async def run_orchestration(input_data: dict = None) -> dict:
    engine = OrchestrationEngine()
    return await engine.execute(input_data)

