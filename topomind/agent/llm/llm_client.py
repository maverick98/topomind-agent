from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Abstract LLM transport interface.

    Responsible only for:
        • Sending prompt
        • Returning raw text
        • Handling backend-specific transport
    """

    @property
    def name(self) -> str:
        """Return backend identity."""
        return self.__class__.__name__

    @abstractmethod
    def chat(self, prompt: str, strict: bool = False) -> str:
        """
        Execute a chat completion and return raw text output.

        Parameters
        ----------
        prompt : str
            Fully constructed prompt.

        strict : bool
            If True, enforce structured JSON behavior
            (e.g., temperature=0, response_format=json).

        Returns
        -------
        str
            Raw model output.
        """
        raise NotImplementedError
