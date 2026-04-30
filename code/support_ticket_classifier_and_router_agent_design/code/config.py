
import os
import logging
from dotenv import load_dotenv

# Load .env file FIRST before any os.getenv() calls
load_dotenv()

class Config:
    _kv_secrets = {}

    # Key Vault secret mapping (only relevant entries for this agent)
    KEY_VAULT_SECRET_MAP = [
        # LLM API keys
        ("AZURE_OPENAI_API_KEY", "openai-secrets.gpt-4.1"),
        ("AZURE_OPENAI_API_KEY", "openai-secrets.azure-key"),
        # Azure Content Safety
        ("AZURE_CONTENT_SAFETY_ENDPOINT", "azure-content-safety-secrets.azure_content_safety_endpoint"),
        ("AZURE_CONTENT_SAFETY_KEY", "azure-content-safety-secrets.azure_content_safety_key"),
        # Observability DB
        ("OBS_AZURE_SQL_SERVER", "agentops-secrets.obs_sql_endpoint"),
        ("OBS_AZURE_SQL_DATABASE", "agentops-secrets.obs_azure_sql_database"),
        ("OBS_AZURE_SQL_PORT", "agentops-secrets.obs_port"),
        ("OBS_AZURE_SQL_USERNAME", "agentops-secrets.obs_sql_username"),
        ("OBS_AZURE_SQL_PASSWORD", "agentops-secrets.obs_sql_password"),
        ("OBS_AZURE_SQL_SCHEMA", "agentops-secrets.obs_azure_sql_schema"),
    ]

    # Models that do not support temperature/max_tokens
    _MAX_TOKENS_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat", "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o3-pro", "o4-mini"
    }
    _TEMPERATURE_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat", "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o3-pro", "o4-mini"
    }

    @classmethod
    def _load_keyvault_secrets(cls):
        """Load secrets from Azure Key Vault if enabled and URI is set."""
        if not getattr(cls, "USE_KEY_VAULT", False):
            return {}
        if not getattr(cls, "KEY_VAULT_URI", ""):
            return {}
        try:
            AZURE_USE_DEFAULT_CREDENTIAL = getattr(cls, "AZURE_USE_DEFAULT_CREDENTIAL", False)
            if AZURE_USE_DEFAULT_CREDENTIAL:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
            else:
                from azure.identity import ClientSecretCredential
                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                client_id = os.getenv("AZURE_CLIENT_ID", "")
                client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
                if not (tenant_id and client_id and client_secret):
                    logging.warning("Service Principal credentials incomplete. Key Vault access will fail.")
                    return {}
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=cls.KEY_VAULT_URI, credential=credential)
            # Group refs by secret name
            by_secret = {}
            for attr, ref in getattr(cls, "KEY_VAULT_SECRET_MAP", []):
                if "." in ref:
                    secret_name, json_key = ref.split(".", 1)
                else:
                    secret_name, json_key = ref, None
                by_secret.setdefault(secret_name, []).append((attr, json_key))
            kv_secrets = {}
            import json as _json
            for secret_name, refs in by_secret.items():
                try:
                    secret = client.get_secret(secret_name)
                    if not secret or not secret.value:
                        logging.debug(f"Key Vault: secret '{secret_name}' is empty or missing")
                        continue
                    raw = secret.value.lstrip('\ufeff')
                    has_json_key = any(json_key is not None for _, json_key in refs)
                    if has_json_key:
                        try:
                            data = _json.loads(raw)
                        except Exception:
                            # Try to repair malformed JSON
                            data = {}
                            try:
                                inner = raw.strip().strip('{}')
                                for part in inner.split(","):
                                    if ':' not in part:
                                        continue
                                    k, v = part.split(':', 1)
                                    k = k.strip().strip('"')
                                    v = v.strip().strip('"')
                                    if k:
                                        data[k] = v
                            except Exception:
                                data = {}
                        if not isinstance(data, dict):
                            continue
                        for attr, json_key in refs:
                            if json_key is not None:
                                val = data.get(json_key)
                                if attr in kv_secrets:
                                    continue
                                if val is not None and val != "":
                                    kv_secrets[attr] = str(val)
                    else:
                        for attr, json_key in refs:
                            if json_key is None and raw:
                                kv_secrets[attr] = raw
                                break
                except Exception as exc:
                    logging.debug(f"Key Vault: failed to fetch secret '{secret_name}': {exc}")
                    continue
            cls._kv_secrets = kv_secrets
            return kv_secrets
        except Exception as exc:
            logging.warning(f"Key Vault: failed to load secrets: {exc}")
            return {}

    @classmethod
    def _validate_api_keys(cls):
        provider = (getattr(cls, "MODEL_PROVIDER", "") or "").lower()
        if provider == "azure":
            if not getattr(cls, "AZURE_OPENAI_API_KEY", ""):
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider.")
        elif provider == "openai":
            if not getattr(cls, "OPENAI_API_KEY", ""):
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        elif provider == "anthropic":
            if not getattr(cls, "ANTHROPIC_API_KEY", ""):
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        elif provider == "google":
            if not getattr(cls, "GOOGLE_API_KEY", ""):
                raise ValueError("GOOGLE_API_KEY is required for Google provider.")

    @classmethod
    def get_llm_kwargs(cls):
        kwargs = {}
        model_lower = (getattr(cls, "LLM_MODEL", "") or "").lower()
        if not any(model_lower.startswith(m) for m in cls._TEMPERATURE_UNSUPPORTED):
            kwargs["temperature"] = getattr(cls, "LLM_TEMPERATURE", None)
        if any(model_lower.startswith(m) for m in cls._MAX_TOKENS_UNSUPPORTED):
            kwargs["max_completion_tokens"] = getattr(cls, "LLM_MAX_TOKENS", None)
        else:
            kwargs["max_tokens"] = getattr(cls, "LLM_MAX_TOKENS", None)
        return kwargs

    @classmethod
    def validate(cls):
        cls._validate_api_keys()

def _initialize_config():
    # Load Key Vault config from .env first
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "").lower() in ("true", "1", "yes")
    KEY_VAULT_URI = os.getenv("KEY_VAULT_URI", "")
    AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")

    setattr(Config, "USE_KEY_VAULT", USE_KEY_VAULT)
    setattr(Config, "KEY_VAULT_URI", KEY_VAULT_URI)
    setattr(Config, "AZURE_USE_DEFAULT_CREDENTIAL", AZURE_USE_DEFAULT_CREDENTIAL)

    # Load Key Vault secrets if enabled
    if USE_KEY_VAULT:
        Config._load_keyvault_secrets()

    # Azure AI Search variables (not used in this agent, but pattern shown for completeness)
    AZURE_SEARCH_VARS = ["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME"]
    AZURE_SP_VARS = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]

    # List of all config variables required by this agent
    CONFIG_VARIABLES = [
        # General
        "ENVIRONMENT",
        # Key Vault config (already loaded above)
        # Azure Service Principal (skip if using DefaultAzureCredential)
        "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET",
        # LLM / Model
        "MODEL_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        # Content Safety
        "AZURE_CONTENT_SAFETY_ENDPOINT", "AZURE_CONTENT_SAFETY_KEY",
        "CONTENT_SAFETY_ENABLED", "CONTENT_SAFETY_SEVERITY_THRESHOLD",
        # Agent identity
        "AGENT_NAME", "AGENT_ID", "PROJECT_NAME", "PROJECT_ID", "SERVICE_NAME", "SERVICE_VERSION",
        # Observability DB
        "OBS_DATABASE_TYPE", "OBS_AZURE_SQL_SERVER", "OBS_AZURE_SQL_DATABASE",
        "OBS_AZURE_SQL_PORT", "OBS_AZURE_SQL_USERNAME", "OBS_AZURE_SQL_PASSWORD",
        "OBS_AZURE_SQL_SCHEMA", "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE",
        # Domain-specific
        "REQUIRE_CONSENT", "VALIDATION_CONFIG_PATH", "VERSION",
        # LLM Models config for cost calculation
        "LLM_MODELS",
    ]

    for var_name in CONFIG_VARIABLES:
        # Skip Service Principal vars if using DefaultAzureCredential
        if var_name in AZURE_SP_VARS and AZURE_USE_DEFAULT_CREDENTIAL:
            continue

        value = None

        # Azure AI Search variables: always from .env (not used here)
        if var_name in AZURE_SEARCH_VARS:
            value = os.getenv(var_name)
        # Standard priority: Key Vault > .env
        elif USE_KEY_VAULT and var_name in Config._kv_secrets:
            value = Config._kv_secrets[var_name]
        else:
            value = os.getenv(var_name)

        # Special handling for booleans
        if var_name in ("CONTENT_SAFETY_ENABLED", "REQUIRE_CONSENT"):
            if value is not None and value != "":
                value = str(value).lower() in ("true", "1", "yes", "on")
            else:
                value = False

        # Special handling for LLM_MODELS (JSON list)
        if var_name == "LLM_MODELS":
            if value:
                try:
                    import json as _json
                    value = _json.loads(value)
                except Exception:
                    logging.warning(f"Invalid JSON for {var_name}: {value}")
                    value = []
            else:
                value = []

        # Convert numeric values to proper types
        if value and var_name == "LLM_TEMPERATURE":
            try:
                value = float(value)
            except ValueError:
                logging.warning(f"Invalid float value for {var_name}: {value}")
        elif value and var_name in ("LLM_MAX_TOKENS", "OBS_AZURE_SQL_PORT"):
            try:
                value = int(value)
            except ValueError:
                logging.warning(f"Invalid integer value for {var_name}: {value}")

        # OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE: default to "yes" if not found
        if var_name == "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
            if not value:
                value = "yes"

        # VERSION: fallback to SERVICE_VERSION if not set
        if var_name == "VERSION" and not value:
            value = os.getenv("SERVICE_VERSION", "")

        # If still not found, warn and set to "" or None
        if value is None or value == "":
            if var_name != "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
                logging.warning(f"Configuration variable {var_name} not found in .env file")
            if var_name in ("LLM_TEMPERATURE", "LLM_MAX_TOKENS", "OBS_AZURE_SQL_PORT"):
                value = None
            elif var_name == "LLM_MODELS":
                value = []
            else:
                value = ""

        setattr(Config, var_name, value)

# Call at module level
_initialize_config()

# Settings instance (backward compatibility with observability module)
settings = Config()
