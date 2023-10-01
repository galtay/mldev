from typing import Optional
from pydantic import BaseModel

class Llama2ChatPrompt(BaseModel):

    B_INST: str = "[INST]"
    E_INST: str = "[/INST]"
    B_SYS: str = "<<SYS>>"
    E_SYS: str = "<</SYS>>"
    BOS: str = "<s>"
    EOS: str = "</s>"

    def get_system_prompt(self, system_message: str, skip_initial_b_seq: bool=False):
        """Note that many tokenizers will add BOS to the front of a tokenized sequence."""
        if skip_initial_b_seq:
            first = ""
        else:
            first = self.BOS
        return f"{first}{self.B_INST} {self.B_SYS}\n{system_message.strip()}\n{self.E_SYS}\n\n"

    def get_prompt(
        self,
        user_message: str,
        system_message: str="",
        response_prefix: Optional[str]=None,
        skip_initial_b_seq: bool=False,
    ) -> str:

        prompt = ""
        system = self.get_system_prompt(system_message, skip_initial_b_seq=skip_initial_b_seq)
        prompt += system
        prompt += f"{user_message.strip()} {self.E_INST}"
        if response_prefix is not None:
            prompt += f" {response_prefix.strip()}"
        return prompt

    def get_conversation_prompt(
        self,
        user_messages: list[str],
        response_messages: list[str],
        system_message: str="",
        response_prefix: Optional[str]=None,
        skip_initial_b_seq: bool=False,
    ) -> str:

        assert len(user_messages) - 1 == len(response_messages)
        prompt = ""
        system = self.get_system_prompt(system_message, skip_initial_b_seq=skip_initial_b_seq)
        prompt += system
        prompt += f"{user_messages[0].strip()} {self.E_INST}"
        for user, response in zip(user_messages[1:], response_messages):
            prompt += f" {response.strip()} {self.EOS}{self.BOS}{self.B_INST} {user.strip()} {self.E_INST}"
        if response_prefix is not None:
            prompt += f" {response_prefix.strip()}"
        return prompt

    def get_inst_only_prompt(self, user_message: str) -> str:
        return f"{self.B_INST} {user_message.strip()} {self.E_INST}"


if __name__ == "__main__":

    l2cp = Llama2ChatPrompt()

    print(l2cp.get_prompt(
        "tell me a story",
        system_message="you are a helpful chatbot",
        response_prefix="Title:"
    ))
    print('-' * 50)


    print(l2cp.get_conversation_prompt(
        ["tell me a story", "good start"],
        response_messages=["once upon a time"],
        system_message="you are a helpful chatbot",
    ))
    print('-' * 50)


    print(l2cp.get_inst_only_prompt("tell me a story"))
    print('-' * 50)
