import json
import os
from openai import OpenAI


GENERATOR_SYSTEM_PROMPT = None


def _load_system_prompt() -> str:
    global GENERATOR_SYSTEM_PROMPT
    if GENERATOR_SYSTEM_PROMPT is None:
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "generator_system.md"
        )
        with open(prompt_path, "r") as f:
            GENERATOR_SYSTEM_PROMPT = f.read()
    return GENERATOR_SYSTEM_PROMPT


def build_generator_user_message(
    cuisine: str,
    count: int,
    search_summary: str,
    feedback: str,
    locked_names: list[str],
    round_num: int,
) -> str:
    parts = [search_summary, ""]

    if feedback:
        parts.append("## IMPROVEMENT FEEDBACK FROM PREVIOUS ROUND")
        parts.append(feedback)
        parts.append("")
        parts.append("## CURRENT STATUS")
        parts.append(f"- {len(locked_names)} proposals already accepted for {cuisine}")
        parts.append(f"- This is round {round_num} of 3")
        parts.append("")
        parts.append("Do NOT repeat these accepted proposals:")
        parts.append(json.dumps(locked_names, indent=2, ensure_ascii=False))
    else:
        parts.append("## FIRST ROUND — no prior feedback. Generate fresh proposals.")

    parts.append("")
    parts.append(f"Generate {count} dish proposals for {cuisine} cuisine.")
    return "\n".join(parts)


class GeneratorAgent:
    def __init__(self, config: dict):
        self.cfg = config["llm"]
        api_key = self.cfg.get("api_key") or os.getenv(self.cfg.get("api_key_env", ""))
        self.client = OpenAI(
            base_url=self.cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            api_key=api_key,
        )
        self.system_prompt = _load_system_prompt()

    def generate(
        self,
        cuisine: str,
        count: int,
        search_summary: str,
        feedback: str = "",
        locked_names: list[str] | None = None,
        round_num: int = 1,
    ) -> list[dict]:
        if locked_names is None:
            locked_names = []

        user_message = build_generator_user_message(
            cuisine=cuisine,
            count=count,
            search_summary=search_summary,
            feedback=feedback,
            locked_names=locked_names,
            round_num=round_num,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.cfg["generator_model"],
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.cfg["generator_temperature"],
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return self._parse_response(content)
        except Exception as e:
            print(f"Generator API call failed: {e}")
            return []

    def _parse_response(self, content: str) -> list[dict]:
        try:
            data = json.loads(content)
            return data.get("proposals", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse Generator response: {e}")
            return []
