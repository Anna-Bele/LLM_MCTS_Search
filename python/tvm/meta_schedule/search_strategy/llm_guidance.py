# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from openai import OpenAI
client = OpenAI()
import time
import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class LLMGuidancePolicy:
    def __init__(
        self,
        model_name: str = "",
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.verbose = verbose

    def pick_mutators(
        self,
        mod,
        available_mutators: List[str],
        historical_perf: Optional[str] = None,
        available_mutator_probs: Optional[Dict[str, float]] = None,
    ) -> Optional[List[str]]:
        tir_text = self._get_tir_as_text(mod)
        system_prompt, user_prompt = self._build_prompt(
            tir_text=tir_text,
            available_mutators=available_mutators,
            historical_perf=historical_perf,
            mutator_probs=available_mutator_probs,
        )
        try:
            logger.warning("You're here inside LLMGuidancePolicy at line 90")
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
            chosen_list = self._extract_mutators_list(content, available_mutators)
            if not chosen_list:
                if self.verbose:
                    logger.warning("LLM did not return a valid list of mutators.")
                return None
            return chosen_list
        except Exception as e:
            logger.warning("OpenAI ChatCompletion failed: %s", str(e))
            return None

    def _get_tir_as_text(self, mod) -> str:
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
            "   and grandparent schedules (their IR, traces, and predicted scores (do not "
            "   overly rely on predicted scores as they can be inaccurate sometimes))\n"
            " - A list of possible mutators (transformations) that can be applied next\n"

            "NOTE: You may repeat the same mutator multiple times if you think it could be "
            "beneficial. For instance, 'meta_schedule.MutateTileSize(...)' might choose "
            "different tile sizes each time, so repeating it can explore a range of tiling"
            "configurations.\n\n"

            "Please compare the IR, trace, and predicted scores of these schedules to "
            "determine what changes might improve the current schedule performance. Then"
            "propose a *sequence* of transformations (one or more) from the provided list."
            "Output the final full mutator name list in the exact format:\n\n"
            "Reasoning: ...\n"
            "Mutators: Fullname1, Fullname2, Fullname3, Fullname4, Fullname5, Fullname6," 
            "Fullname7, Fullname8, Fullname9, Fullname10...\n\n"
            "Only output them in a single final line starting with 'Mutators:'.\n"
            "This will allow MCTS to explore those transformations.\n"

            "IMPORTANT: If you choose one of the mutators, you MUST include the number in"
            "parentheses following the mutator"
            "If you omit the '(0x...)' part, your answer is invalid.\n\n"
        )

        user_msg = (
            f"=== IRModule (TensorIR) for the current schedule ===\n"
            f"```python\n{tir_text}\n```\n\n"
        )

        if historical_perf:
            user_msg += (
                "=== Historical Performance Info (Leaf, Parent, Grandparent) ===\n"
                f"{historical_perf}\n\n"
            )

        user_msg += (
            "=== Available Mutators ===\n"
            f"{available_mutators}\n\n"
        )

        user_msg += (
            "Please analyze the IR, the transformations (Trace), and the predicted scores "
            "across the current schedule and its ancestors. Then propose a *sequence* "
            "of mutators (from the provided list) that you believe will improve performance. "
            "Remember: You can reuse the same mutator multiple times if that seems helpful. "
            "Use chain-of-thought reasoning to explain your logic step by step, and then "
            "output the the recommended transformations (full mutator names) in a single line"
            "labeled 'Mutators: Fullname1, Fullname2, Fullname3, Fullname4, Fullname5" 
            "Fullname6, Fullname7, Fullname8, Fullname9, Fullname10...'.\n\n"
            "Example:\n\n"
            "Reasoning: This schedule still has large loop extents, so I'd tile it twice" 
            "differently, then unroll...\n"
            "Mutators: meta_schedule....(...), meta_schedule....(...), meta_schedule....(...),"
            "meta_schedule....(...), meta_schedule....(...), ...\n"
            "IMPORTANT: If you choose one of the mutators, you MUST include the number in" 
            "parentheses following the mutator."
            "If you omit the '(0x...)' part, your answer is invalid.\n\n"
        )

        return system_msg, user_msg

    def _extract_mutators_list(
        self,
        model_text: str,
        valid_mutators: List[str]
    ) -> List[str]:
        chosen_list = []
        for line in model_text.splitlines():
            line = line.strip()
            if line.lower().startswith("mutators:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    # everything after the colon
                    remainder = parts[1].strip()
                    raw_names = [r.strip() for r in remainder.split(",")]
                    for candidate in raw_names:
                        if candidate in valid_mutators:
                            chosen_list.append(candidate)
                break
        return chosen_list