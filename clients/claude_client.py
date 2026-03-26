import json
import anthropic

import config


class ClaudeClient:
    def __init__(self, api_key: str = config.CLAUDE_API_KEY):
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, prompt: str, system: str = "", model: str = config.MODEL_REPORT) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": model, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text

    def chat_json(self, prompt: str, system: str = "", model: str = config.MODEL_REPORT) -> dict:
        raw = self.chat(prompt, system=system, model=model)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
