from openai import OpenAI
client = OpenAI()
import time
import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class LLMGuidancePolicy:
    """
    A policy that uses an LLM (e.g., GPT-4) to pick the next mutator action
    by analyzing:
      - The IRModule (TIR code),
      - Optional summarized performance info or prior transformations (historical_perf),
      - A list of possible mutators, and
      - their probabilities.

    Usage:
      1) Instantiate with your OpenAI API key (and optional parameters).
      2) Call pick_mutator(...) each time you want the LLM to choose the next transformation.
    """

    def __init__(
        self,
        # openai_api_key: str,
        model_name: str = "gpt-4o-mini",
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        openai_api_key : str
            Your OpenAI API key. For instance, os.environ["OPENAI_API_KEY"].

        model_name : str
            The name of the model to use, e.g. 'gpt-4', 'gpt-3.5-turbo', etc.

        temperature : float
            The temperature for OpenAI's completion sampling (0.0 => deterministic).

        max_tokens : int
            The maximum number of tokens to generate in each LLM response.

        verbose : bool
            If True, print out debug info and the raw LLM responses.
        """
        # self.api_key = openai_api_key
        # openai.api_key = self.api_key

        self.model_name = model_name
        self.verbose = verbose

    def pick_mutator(
        self,
        mod,
        available_mutators: List[str],
        historical_perf: Optional[str] = None,
        available_mutator_probs: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Query the LLM to pick the next mutator. This builds a prompt that includes:
          - TIR code from 'mod' (the IRModule)
          - Any historical performance info in 'historical_perf' 
            (e.g. leaf, immediate parent, grandparent schedules)
          - A list of valid mutator names ('available_mutators')
          - Mutator probabilities ('available_mutator_probs')

        The LLM is expected to compare the IR, trace, and predicted scores
        of the current schedule, immediate parent schedule, and grandparent schedule
        (if given), to see what changes can potentially improve performance.

        Returns
        -------
        chosen_mutator : Optional[str]
            The name of the mutator (e.g. "Parallelize" or "Unroll") 
            that the LLM recommends, or None if invalid/fail.
        """
        # 1) Convert IRModule to TIR text
        tir_text = self._get_tir_as_text(mod)

        # 2) Build system/user prompts
        system_prompt, user_prompt = self._build_prompt(
            tir_text=tir_text,
            available_mutators=available_mutators,
            historical_perf=historical_perf,
            mutator_probs=available_mutator_probs,
        )

        try:
            logger.warning("You're here inside LLMGuidancePolicy at line 90")
            # 3) Call OpenAI ChatCompletion
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            logger.warning("You're here inside LLMGuidancePolicy at line 99")

            content = response.choices[0].message.content
            if self.verbose:
                logger.warning("LLM raw response:\n%s", content)

            # 4) Parse the model's response to find "Mutator: X"
            chosen = self._extract_mutator_name(content, available_mutators)
            if chosen is None:
                if self.verbose:
                    logger.warning("LLM did not return a valid mutator from the list.")
                return None
            return chosen

        except Exception as e:
            logger.warning("OpenAI ChatCompletion failed: %s", str(e))
            return None

    def _get_tir_as_text(self, mod) -> str:
        """
        Convert the IRModule to text. Usually mod.script(show_meta=True) is fine.
        You can customize if it's too large or if you want partial printing.
        """
        try:
            return mod.script(show_meta=True)
        except Exception as e:
            logger.warning("Failed to script the IRModule: %s", str(e))
            return "<IRModule scripting failed>"

    def _build_prompt(
        self,
        tir_text: str,
        available_mutators: List[str],
        historical_perf: Optional[str],
        mutator_probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, str]:
        """
        Construct the system and user prompts for the LLM. 
        The system prompt sets the role & final output format requirement.
        The user prompt contains:
          - TIR code
          - optional performance data (leaf + parent + grandparent info)
          - the list of mutators
          - optionally mutator probabilities

        We also describe the MCTS context, 
        instructing the LLM to compare the IR, trace, predicted scores
        among the schedules, then choose the best next mutator.
        """

        # Provide more detailed background in the system message
        system_msg = (
            "You are an AI scheduling assistant integrated with TVM MetaSchedule. "
            "We are performing a Monte Carlo Tree Search (MCTS) to find an optimal "
            "schedule transformation sequence for a given IRModule. In this MCTS tree, "
            "the 'current schedule' is the leaf we are expanding, while 'immediate parent' "
            "and 'grandparent' refer to the ancestors in the tree. Each schedule has an IR, "
            "a sequence of transformations (the trace), and a predicted performance score "
            "from TVM's default XGBoost cost model.\n\n"
            "You are given:\n"
            " - The IRModule for the current schedule\n"
            " - Historical performance info summarizing the current schedule, its parent, "
            "   and grandparent schedules (their IR, traces, and predicted scores)\n"
            " - A list of possible mutators (transformations) that can be applied next\n"
            " - Mutator probabilities from the internal policy, indicating how "
            "   likely each mutator is under normal random selection.\n\n"
            "Please compare the IR, trace, and predicted scores of these schedules to see what "
            "changes might improve the predicted performance. Then pick exactly one mutator "
            "from the provided list. You must output your reasoning as well as the final mutator "
            "in the format:\n\n"
            "Reasoning: ...\n"
            "Mutator: <name>\n\n"
            "This will allow MCTS to explore that transformation.\n"
        )

        # The user prompt: includes IR, performance info, list of mutators, and instructions
        user_msg = (
            f"=== IRModule (TensorIR) for the current schedule ===\n"
            f"```python\n{tir_text}\n```\n\n"
        )

        # If there's historical performance data, describing current+parent+grandparent
        # with IR, traces, and predicted scores, we display it:
        if historical_perf:
            user_msg += (
                "=== Historical Performance Info (Leaf, Parent, Grandparent) ===\n"
                f"{historical_perf}\n\n"
            )

        # The list of available mutators
        user_msg += (
            "=== Available Mutators ===\n"
            f"{available_mutators}\n\n"
        )

        # If we have mutator probabilities, display them
        if mutator_probs:
            user_msg += (
                "=== Mutator Probabilities ===\n"
                "Below are the probabilities assigned by our internal policy. A higher probability "
                "suggests the mutator is more likely under normal random selection:\n"
            )
            for name, prob in mutator_probs.items():
                user_msg += f"  - {name}: {prob}\n"
            user_msg += "\n"

        # Now the final instruction
        user_msg += (
            "Please analyze the IR, the transformations (Trace), and the predicted scores "
            "across the current schedule and its ancestors. Then decide which single "
            "mutator (from the provided list) would likely improve the schedule. "
            "Use chain-of-thought reasoning to explain your logic step by step, and then "
            "output the chosen mutator in exactly one final line:\n\n"
            "Mutator: <one_of_the_list>\n\n"
            "Example:\n\n"
            "Reasoning: The loop is large and unrolled, so I'd like to tile it...\n"
            "Mutator: Tile\n"
        )

        return system_msg, user_msg

    def _extract_mutator_name(
        self,
        model_text: str,
        valid_mutators: List[str]
    ) -> Optional[str]:
        """
        Scan the LLM's output to find 'Mutator: XX', 
        then confirm 'XX' is in valid_mutators. If not found, return None.
        """
        chosen = None
        for line in model_text.splitlines():
            line = line.strip()
            if line.lower().startswith("mutator:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    candidate = parts[1].strip()
                    if candidate in valid_mutators:
                        chosen = candidate
                        break
        return chosen
