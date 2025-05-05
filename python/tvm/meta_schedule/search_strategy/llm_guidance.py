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

    def pick_mutators(
        self,
        mod,
        available_mutators: List[str],
        historical_perf: Optional[str] = None,
        available_mutator_probs: Optional[Dict[str, float]] = None,
    ) -> Optional[List[str]]:
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

        best_trace_str = """
        ["b0 = sch.get_block(name="matmul", func_name="main")", 
        "b1 = sch.get_block(name="root", func_name="main")", 
        "sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")", 
        "l2, l3, l4, l5 = sch.get_loops(block=b0)", 
        "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64, decision=[1, 1, 1, 1])", 
        "l10, l11, l12, l13 = sch.split(loop=l2, factors=[v6, v7, v8, v9], preserve_unit_iters=True, disable_predication=False)", 
        "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64, decision=[2, 1, 2, 4])", 
        "l18, l19, l20, l21 = sch.split(loop=l3, factors=[v14, v15, v16, v17], preserve_unit_iters=True, disable_predication=False)", 
        "v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64, decision=[2, 1, 2048, 1])", 
        "l26, l27, l28, l29 = sch.split(loop=l4, factors=[v22, v23, v24, v25], preserve_unit_iters=True, disable_predication=False)", 
        "v30, v31 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64, decision=[1024, 4])", 
        "l32, l33 = sch.split(loop=l5, factors=[v30, v31], preserve_unit_iters=True, disable_predication=False)", 
        "sch.reorder(l10, l18, l26, l11, l19, l27, l32, l12, l20, l28, l33, l13, l21, l29)", 
        "b34 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")", 
        "sch.reverse_compute_at(block=b34, loop=l27, preserve_unit_loops=True, index=-1)", 
        "sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=16)", 
        "sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)", 
        "v35 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0)", 
        "sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v35)", "sch.enter_postproc()", 
        "b36 = sch.get_block(name="root", func_name="main")", 
        "sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.parallel")", 
        "sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.vectorize")", 
        "sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_explicit")", 
        "b37, b38 = sch.get_child_blocks(b36)", "l39, l40, l41, l42, l43, l44, l45, l46, l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b37)", 
        "l53 = sch.fuse(l39, l40, l41, l42, l43, l44, preserve_unit_iters=True)", 
        "sch.parallel(loop=l53)", "l54, l55, l56, l57 = sch.get_loops(block=b38)", 
        "l58 = sch.fuse(l54, preserve_unit_iters=True)", "sch.parallel(loop=l58)", 
        "b59 = sch.get_block(name="matmul", func_name="main")", 
        "l60, l61, l62, l63, l64, l65, l66, l67, l68 = sch.get_loops(block=b59)", 
        "b69 = sch.decompose_reduction(block=b59, loop=l61)"]
        """

        best_ir_str = """
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((1, 16, 4096), "float32"), B: T.Buffer((4096, 4096), "float32"), C: T.Buffer((1, 16, 4096), "float32")):
                T.func_attr({"tir.noalias": T.bool(True)})
                # with T.block("root"):
                C_global = T.alloc_buffer((1, 16, 4096))
                for b_0_i_0_j_0_b_1_i_1_j_1_fused_fused in T.parallel(4):
                    for b_2_init, i_2_init, j_2_init, b_3_init, i_3_init, j_3_init in T.grid(1, 2, 2048, 1, 4, 1):
                        with T.block("matmul_init"):
                            vb = T.axis.spatial(1, b_2_init + b_3_init)
                            vi = T.axis.spatial(16, b_0_i_0_j_0_b_1_i_1_j_1_fused_fused // 2 * 8 + i_2_init * 4 + i_3_init)
                            vj = T.axis.spatial(4096, b_0_i_0_j_0_b_1_i_1_j_1_fused_fused % 2 * 2048 + j_2_init + j_3_init)
                            T.reads()
                            T.writes(C_global[vb, vi, vj])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            C_global[vb, vi, vj] = T.float32(0.0)
                    for k_0, b_2, i_2, j_2, k_1, b_3, i_3, j_3 in T.grid(1024, 1, 2, 2048, 4, 1, 4, 1):
                        with T.block("matmul_update"):
                            vb = T.axis.spatial(1, b_2 + b_3)
                            vi = T.axis.spatial(16, b_0_i_0_j_0_b_1_i_1_j_1_fused_fused // 2 * 8 + i_2 * 4 + i_3)
                            vj = T.axis.spatial(4096, b_0_i_0_j_0_b_1_i_1_j_1_fused_fused % 2 * 2048 + j_2 + j_3)
                            vk = T.axis.reduce(4096, k_0 * 4 + k_1)
                            T.reads(C_global[vb, vi, vj], A[vb, vi, vk], B[vk, vj])
                            T.writes(C_global[vb, vi, vj])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            C_global[vb, vi, vj] = C_global[vb, vi, vj] + A[vb, vi, vk] * B[vk, vj]
                    for ax0, ax1, ax2 in T.grid(1, 8, 2048):
                        with T.block("C_global"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(16, b_0_i_0_j_0_b_1_i_1_j_1_fused_fused // 2 * 8 + ax1)
                            v2 = T.axis.spatial(4096, b_0_i_0_j_0_b_1_i_1_j_1_fused_fused % 2 * 2048 + ax2)
                            T.reads(C_global[v0, v1, v2])
                            T.writes(C[v0, v1, v2])
                            C[v0, v1, v2] = C_global[v0, v1, v2]
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

            "We also provide you with a best-performing trace and IR from a previous run. "
            "Feel free to refer to them if they can guide you to propose better transformations.\n\n"

            "You are given:\n"
            " - The IRModule for the current schedule\n"
            " - Historical performance info summarizing the current schedule, its parent, "
            "   and grandparent schedules (their IR, traces, and predicted scores)\n"
            " - A best-performing trace & IR from earlier runs\n"
            " - A list of possible mutators (transformations) that can be applied next\n"
            " - Mutator probabilities from the internal policy, indicating how "
            "   likely each mutator is under normal random selection.\n\n"

            "NOTE: You may repeat the same mutator multiple times if you think it could be beneficial. "
            "For instance, 'meta_schedule.MutateTileSize(...)' might choose different tile sizes each time, "
            "so repeating it can explore a range of tiling configurations.\n\n"

            "Please compare the IR, trace, predicted scores of these schedules and the best-performing artifacts to see what "
            "changes might improve the predicted performance. Then propose a *sequence* of transformations "
            "(one or more) from the provided list. Output your chain-of-thought reasoning as well as "
            "the final full mutator name list in the exact format:\n\n"
            "Reasoning: ...\n"
            "Mutators: Fullname1, Fullname2, Fullname3, Fullname4, Fullname5, ...\n\n"
            "Only output them in a single final line labeled 'Mutators:'.\n"
            "This will allow MCTS to explore those transformations.\n"

            "IMPORTANT: If you choose one of the mutators, you MUST include the number in parentheses following the mutator"
            "If you omit the '(0x...)' part, your answer is invalid.\n\n"
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

        user_msg += (
            "=== Best-Performing Trace (from previous runs) ===\n"
            f"{best_trace_str}\n\n"
        )
        user_msg += (
            "=== Best-Performing IR (from previous runs) ===\n"
            f"```python\n{best_ir_str}\n```\n\n"
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
            "across the current schedule and its ancestors. Then propose a *sequence* "
            "of mutators (from the provided list) that you believe will improve performance. "
            "Remember: You can reuse the same mutator multiple times if that seems helpful. "
            "Use chain-of-thought reasoning to explain your logic step by step, and then "
            "output the the recommended transformations (full mutator names) in a single line labeled 'Mutators: Fullname1, Fullname2, Fullname3, Fullname4, Fullname5...'.\n\n"
            "Example:\n\n"
            "Reasoning: This schedule still has large loop extents, so I'd tile it twice differently, then unroll...\n"
            "Mutator: meta_schedule....(...), meta_schedule....(...), meta_schedule....(...), meta_schedule....(...), meta_schedule....(...), ...\n"
            "IMPORTANT: If you choose one of the mutators, you MUST include the number in parentheses following the mutator."
            "If you omit the '(0x...)' part, your answer is invalid.\n\n"
        )

        return system_msg, user_msg

    def _extract_mutators_list(
        self,
        model_text: str,
        valid_mutators: List[str]
    ) -> List[str]:
        """
        Scan the LLM's output to find 'Mutator: XX', 
        then confirm 'XX' is in valid_mutators. If not found, return None.
        """
        chosen_list = []
        for line in model_text.splitlines():
            line = line.strip()
            if line.lower().startswith("mutators:"):
                # e.g. "Mutators: parallelize, unroll"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    # everything after the colon
                    remainder = parts[1].strip()
                    # split by comma
                    raw_names = [r.strip() for r in remainder.split(",")]
                    # filter valid
                    for candidate in raw_names:
                        if candidate in valid_mutators:
                            chosen_list.append(candidate)
                break  # only parse the first 'Mutators:' line we see
        return chosen_list
    
        # chosen = None
        # for line in model_text.splitlines():
        #     line = line.strip()
        #     if line.lower().startswith("mutator:"):
        #         parts = line.split(":", 1)
        #         if len(parts) == 2:
        #             candidate = parts[1].strip()
        #             if candidate in valid_mutators:
        #                 chosen = candidate
        #                 break
        # return chosen
