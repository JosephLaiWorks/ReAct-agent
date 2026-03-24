import os
import re
from groq import Groq
from dotenv import load_dotenv
from tools import search_web

load_dotenv()


class ReActAgent:
    def __init__(self, system_prompt: str, max_steps: int = 5, model: str = "llama-3.1-8b-instant"):
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.model = model
        self.history = []
        self.searched_queries = []

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=api_key)

    def build_messages(self, query: str):
        conversation = [f"Question: {query}"]
        if self.history:
            conversation.extend(self.history)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(conversation)},
        ]

    def normalize_query(self, q: str) -> str:
        return re.sub(r"\s+", " ", q.strip().lower())

    def tokenize_query(self, q: str):
        return set(re.findall(r"[a-z0-9]+", q.lower()))

    def is_near_duplicate_query(self, q: str) -> bool:
        current_norm = self.normalize_query(q)
        current_tokens = self.tokenize_query(q)

        refinement_tokens = {"total", "current", "exact", "difference", "vs", "versus"}

        for old_q in self.searched_queries:
            old_norm = self.normalize_query(old_q)
            old_tokens = self.tokenize_query(old_q)

            if current_norm == old_norm:
                return True

            new_tokens = current_tokens - old_tokens
            if new_tokens & refinement_tokens:
                continue

            if current_tokens and old_tokens:
                overlap = len(current_tokens & old_tokens)
                similarity = overlap / max(len(current_tokens), len(old_tokens))
                if similarity >= 0.9:
                    return True

        return False

    def is_arithmetic_query(self, q: str) -> bool:
        q = q.strip()
        if not q:
            return False
        allowed_chars = set("0123456789.,+-*/()% ")
        has_operator = any(op in q for op in ["+", "-", "*", "/", "%"])
        return has_operator and all(ch in allowed_chars for ch in q)

    def extract_first_valid_block(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text

        final_match = re.search(
            r"Thought:\s*(.*?)\nFinal Answer:\s*(.*?)(?=\n(?:Thought:|Action:|Final Answer:|Observation:)|\Z)",
            text,
            re.DOTALL,
        )
        if final_match and final_match.start() == 0:
            thought = re.sub(r"\s+", " ", final_match.group(1)).strip()
            answer = re.sub(r"\s+", " ", final_match.group(2)).strip()
            return f"Thought: {thought}\nFinal Answer: {answer}"

        action_match = re.search(
            r'Action:\s*Search\["(.*?)"\]',
            text,
            re.DOTALL,
        )
        if text.startswith("Thought:") and action_match:
            thought_part = text[:action_match.start()].strip()
            thought_part = re.sub(r"^Thought:\s*", "", thought_part, flags=re.DOTALL).strip()
            thought = re.sub(r"\s+", " ", thought_part)
            action_query = action_match.group(1).strip()
            return f'Thought: {thought}\nAction: Search["{action_query}"]'

        return text

    def call_llm(self, query: str) -> str:
        messages = self.build_messages(query)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                stop=["\nObservation:"],
            )
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "rate_limit" in error_text.lower():
                return (
                    "Thought: I hit a rate limit and cannot continue searching right now.\n"
                    "Final Answer: I could not finish because the model API hit a rate limit. Please rerun later or switch to a lighter model."
                )
            return (
                "Thought: The LLM call failed unexpectedly.\n"
                f"Final Answer: I could not continue because of an API error: {error_text}"
            )

        content = response.choices[0].message.content or ""
        return self.extract_first_valid_block(content)

    def parse_action(self, text: str):
        match = re.fullmatch(
            r'Thought:\s*.*?\nAction:\s*Search\["(.*?)"\]',
            text.strip(),
            re.DOTALL,
        )
        return match.group(1).strip() if match else None

    def parse_final_answer(self, text: str):
        match = re.fullmatch(
            r"Thought:\s*.*?\nFinal Answer:\s*(.+)",
            text.strip(),
            re.DOTALL,
        )
        return match.group(1).strip() if match else None

    def final_answer_is_incomplete(self, answer: str) -> bool:
        lowered = answer.lower().strip()
        incomplete_markers = [
            "i need",
            "need to know",
            "not directly stated",
            "not enough information",
            "insufficient information",
            "cannot determine",
            "can't determine",
            "could not determine",
            "i do not know",
            "i don't know",
            "unknown",
            "to find",
            "do not clearly state",
            "does not clearly state",
            "do not clearly provide",
            "does not clearly provide",
            "not clearly provide",
            "not clearly state",
            "unfortunately",
            "existing observations do not",
            "the observations do not",
        ]
        if any(marker in lowered for marker in incomplete_markers):
            return True

        if len(lowered) < 8:
            return True

        return False

    def invalid_format_observation(self) -> str:
        return (
            "Observation:\n"
            "Invalid format. Output exactly one block only: "
            "either Thought followed by Action: Search[\"...\"] "
            "or Thought followed by Final Answer: .... "
            "Do not output Action: None."
        )

    def run(self, query: str):
        self.history = []
        self.searched_queries = []

        for step in range(1, self.max_steps + 1):
            print(f"\n--- Step {step} ---")

            llm_output = self.call_llm(query)
            print(llm_output)

            has_action = self.parse_action(llm_output) is not None
            has_final = self.parse_final_answer(llm_output) is not None

            if has_action and has_final:
                observation = self.invalid_format_observation()
                print(observation)
                self.history.append(observation)
                continue

            if llm_output.count('Search["') > 1:
                observation = self.invalid_format_observation()
                print(observation)
                self.history.append(observation)
                continue

            final_answer = self.parse_final_answer(llm_output)
            if final_answer is not None:
                if self.final_answer_is_incomplete(final_answer):
                    observation = (
                        "Observation:\n"
                        "That Final Answer is incomplete because it does not actually answer the question. "
                        "If a required fact is still missing, do not give up. Reflect and try one genuinely new, more specific search for that missing fact "
                        "(for example by adding words like total, current, or exact year), then continue the ReAct loop."
                    )
                    print(observation)
                    self.history.append(llm_output)
                    self.history.append(observation)
                    continue
                self.history.append(llm_output)
                return final_answer

            action_query = self.parse_action(llm_output)
            if action_query is None:
                observation = self.invalid_format_observation()
                print(observation)
                self.history.append(observation)
                continue

            self.history.append(llm_output)

            if self.is_arithmetic_query(action_query):
                observation = (
                    "Observation:\n"
                    "This query is only arithmetic. Do not search for arithmetic. Use the numbers already found and give the Final Answer directly."
                )
                print(observation)
                self.history.append(observation)
                continue

            if self.is_near_duplicate_query(action_query):
                observation = (
                    "Observation:\n"
                    "This search is identical or nearly identical to one you already used. Do not repeat it. "
                    "If a required fact is still missing, keep the same entity and year but make the query more specific by adding the missing attribute "
                    '(for example: total, current, exact value, population, refresh rate, founder, or CEO).'
                )
                print(observation)
                self.history.append(observation)
                continue

            self.searched_queries.append(action_query)
            tool_result = search_web(action_query)
            observation = f"Observation:\n{tool_result}"
            print(observation)
            self.history.append(observation)

        return "Failed: Reached max steps without a final answer."
