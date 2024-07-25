"""Defines class that uses magic commands to interact with the `ConversableAgent`."""

import io
from contextlib import redirect_stdout

from autogen import ConversableAgent, register_function
from IPython.core.magic import Magics, cell_magic, magics_class

from msticpy.aiagents.config_utils import get_autogen_config_from_msticpyconfig
from msticpy.aiagents.rag_agent import add_two_numbers


@magics_class
class CodeAutogenMagic(Magics):
    """
    A class that uses magic commands in IPython to interact with the ConversableAgent.

    Attributes
    ----------
    assistant_instance : ConversableAgent
        The assistant agent that suggests the tool calls.
    user_proxy : ConversableAgent
        The user proxy agent that interacts with the assistant agent and executes the tool calls.
    """

    def __init__(self, shell):
        """
        Construct all the necessary attributes for the CodeAutogenMagic object.

        Parameters
        ----------
        shell : InteractiveShell
            The shell in which to register the magic commands.
        """
        super().__init__(shell)
        # The assistant agent suggests the tool calls
        self.assistant_instance = ConversableAgent(
            name="Assistant",
            system_message="""You are a helpful AI Assistant that can help with MSTICpy functions.
            Return 'TERMINATE' when the task is done.""",
            llm_config=get_autogen_config_from_msticpyconfig(),
        )

        # The user proxy agent interacts with the assistant agent and executes the tool calls
        self.user_proxy = ConversableAgent(
            name="User",
            llm_config=False,
            is_termination_msg=lambda msg: msg.get("content") is not None
            and "TERMINATE" in msg["content"],
            human_input_mode="NEVER",
        )

        register_function(
            add_two_numbers,
            caller=self.assistant_instance,  # Assistant agent suggests calls to the function
            executor=self.user_proxy,  # User proxy agent executes the function calls
            name="add_two_numbers",  # Default: Function name is used as the tool name
            description="Adds numbers a and b",
        )

    def code_magic(self, question: str) -> str:
        """
        Initiate assistant chat, prints the question and answer, and returns the chat summary.

        Parameters
        ----------
        question : str
            The question for the assistant.

        Returns
        -------
        str
            The chat summary.

        Example
        -------
        >>> code_magic("What is 3500000+28492739?")
        """
        self.assistant_instance.reset()
        output = io.StringIO()
        with redirect_stdout(output):
            chat_response = self.user_proxy.initiate_chat(
                self.assistant_instance,
                message=question,
                summary_method="reflection_with_llm",
            )

        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {chat_response.summary}")

        return chat_response.summary

    @cell_magic
    def code(self, cell: str):
        """
        Ask the agent to write Python code.

        Parameters
        ----------
        cell : str
            The content of the cell. This is used as the question to ask the agents.

        Example Usage
        -------------
        To ask the agents to write a Python code snippet, use the cell magic
        command followed by the question in the cell.
        For example:

        %%code
        What is 3500000+28492739?

        """
        self.code_magic(cell)


# Register the magic class with IPython
def load_ipython_extension(ipython):
    """Register the magic class with IPython."""
    ipython.register_magics(CodeAutogenMagic)
