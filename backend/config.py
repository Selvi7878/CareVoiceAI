"""
Shared configuration — reads your .env and provides a reusable
Azure OpenAI chat client using API key auth (not DefaultAzureCredential).

Maps your existing v1 env var names to what the code needs.
"""

import os
from agent_framework.azure import AzureOpenAIChatClient
from azure.core.credentials import AzureKeyCredential


def get_openai_client() -> AzureOpenAIChatClient:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ.get(
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
    )

    return AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment,
        api_key=api_key,
    )


def get_cosmos_database() -> str:
    return os.environ.get("AZURE_COSMOS_DATABASE", os.environ.get("COSMOS_DATABASE", "carevoice"))


def get_cosmos_container() -> str:
    return os.environ.get("AZURE_COSMOS_CONTAINER", "patients")
