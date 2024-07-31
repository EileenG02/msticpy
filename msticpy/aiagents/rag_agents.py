# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Module for MSTICpy documentation utilities and retrieval agent configuration.

Includes functions to find documentation files and to set up retrieval
agents that assist security analysts by answering questions based on MSTICpy documentation.
"""

import importlib.resources as pkg_resources
from pathlib import Path
from typing import Optional

from autogen.agentchat.chat import ChatResult
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

from .._version import VERSION
from .config_utils import get_autogen_config_from_msticpyconfig

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.

    Usage: add_two_numbers(a, b)
    """
    return a + b

def find_rst_files():
    """
    Find all .rst files in the docs/source directory of 'msticpy' package.

    Returns
    -------
    list of str
        List of paths to .rst files in the docs/source directory.
    """
    # Get the path to the docs/source directory of the package
    docs_path = Path(pkg_resources.files("msticpy")).parent / "docs" / "source"

    # Find all .rst files in the docs/source directory
    rst_files = list(str(fp) for fp in docs_path.rglob("*.rst"))

    return rst_files


def get_retrieval_assistant_agent(system_message: str = "") -> RetrieveAssistantAgent:
    """
    Create and return a RetrieveAssistantAgent.

    Parameters
    ----------
    system_message : str, optional
        Custom system message for the assistant.

    Returns
    -------
    RetrieveAssistantAgent
        Configured RetrieveAssistantAgent instance.
    """
    if not system_message:
        system_message = (
            "You are a helpful assistant to security analysts using MSTICpy."
        )
    return RetrieveAssistantAgent(
        name="assistant",
        system_message=system_message,
        llm_config=get_autogen_config_from_msticpyconfig(),
    )


def get_retrieval_user_proxy_agent(
    customized_prompt: Optional[str] = None,
) -> RetrieveUserProxyAgent:
    """
    Create and return a RetrieveUserProxyAgent.

    Parameters
    ----------
    customized_prompt : Optional[str], optional
        Custom prompt for the assistant agent, by default None.

    Returns
    -------
    RetrieveUserProxyAgent
        Configured RetrieveUserProxyAgent instance.
    """
    rst_files = find_rst_files()
    autogen_config = get_autogen_config_from_msticpyconfig()
    return RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: True,
        retrieve_config={
            "task": "default",
            "docs_path": rst_files,
            "chunk_token_size": 2000,
            "customized_prompt": customized_prompt,
            "model": autogen_config["config_list"][0]["model"],
            "vector_db": "chroma",
            "collection_name": f"MSTICpy_Docs_{VERSION}",
            "get_or_create": True,
        },
        code_execution_config=False,
    )


def ask_question(
    assistant_agent: RetrieveAssistantAgent,
    user_proxy_agent: RetrieveUserProxyAgent,
    question: str,
) -> ChatResult:
    """
    Ask a question using the assistant and user proxy agents.

    Parameters
    ----------
    assistant_agent : RetrieveAssistantAgent
        The assistant agent to use.
    user_proxy_agent : RetrieveUserProxyAgent
        The user proxy agent to use.
    question : str
        The question to ask.

    Returns
    -------
    ChatResult
        The result of the chat interaction.
    """
    assistant_agent.reset()
    return user_proxy_agent.initiate_chat(
        assistant_agent, message=user_proxy_agent.message_generator, problem=question
    )
