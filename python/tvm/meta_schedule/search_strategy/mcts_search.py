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

import math
import random
import logging
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
    Dict,
    Optional,
    Set,
)

import tvm
from tvm._ffi import get_global_func
from tvm.runtime import Object
from tvm.tir import Schedule
from tvm.tir.schedule import Trace
from tvm.ir import IRModule

from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.runner import RunnerResult
from .search_strategy import SearchStrategy
from .search_strategy import PySearchStrategy
from .search_strategy import MeasureCandidate
from ..postproc import Postproc
from ..mutator import Mutator
from ..database import Workload
from .. import _ffi_api
from .llm_guidance import LLMGuidancePolicy

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext

from ..database import TuningRecord

try:
    from tvm.error import InvalidScheduleError
except ImportError:
    InvalidScheduleError = tvm.TVMError

logger = logging.getLogger("meta_schedule")
logger.setLevel(logging.DEBUG)


class _SizedMinHeap:
    def __init__(self, size_limit: int):
        self._size_limit = size_limit
        self._heap = []
        self._push_counter = 0  # strictly increasing for tie-breaking

    def push(self, sch: Schedule, score: float, measured_flag: bool) -> None:
        neg_score = -score
        self._push_counter += 1
        item = (neg_score, self._push_counter, sch, measured_flag)
        if len(self._heap) < self._size_limit:
            heapq.heappush(self._heap, item)
        else:
            worst_neg, _, _, _ = self._heap[0]
            if neg_score > worst_neg:
                return
            heapq.heapreplace(self._heap, item)

    def items_descending(self) -> List[Tuple[float, Schedule, bool]]:
        items = []
        for (neg, _, sch, meas) in self._heap:
            score = -neg
            items.append((score, sch, meas))
        items.sort(key=lambda x: x[0], reverse=True)
        return items


class MCTSNode:
    __slots__ = [
        "schedule",
        "parent",
        "children",
        "visits",
        "total_value",
        "depth",
    ]

    def __init__(
        self,
        schedule: Optional[Schedule],
        parent: Optional["MCTSNode"],
        depth: int,
    ):
        self.schedule = schedule
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.total_value = 0.0
        self.depth = depth

    def clone_tree(self) -> "MCTSNode":
        new_node = MCTSNode(self.schedule, None, self.depth)
        new_node.visits = self.visits
        new_node.total_value = self.total_value
        for ch in self.children:
            child_copy = ch.clone_tree()
            child_copy.parent = new_node
            new_node.children.append(child_copy)
        return new_node


class MCTSTuner:
    """
    Implements the core Monte Carlo Tree Search routines including but not limited to:
      - UCB-based node selection
      - expansions
      - rollouts/simulations
      - backprop
      - cost model predictions
    """

    def __init__(
        self,
        population_size: int,
        init_measured_ratio: float,
        init_min_unmeasured: int,
        max_fail_count: int,
        genetic_num_iters: int,
        genetic_mutate_prob: float,
        genetic_max_fail_count: int,
        num_empty_iters_before_early_stop: int,
        max_stale_iters: int,
        diversity_epsilon: float,
        max_stale_diversity_iters: int,
        trace_commit: bool,
        verbose: int,
        mcts_ucb_constant: float,
        mcts_max_depth: Optional[int],
        mcts_num_threads: int,
        mcts_num_rollouts_per_expansion: int,
        # references
        postprocs: List[Postproc],
        mutator_probs: Dict[Mutator, float],
        context: "TuneContext",
        cost_model: Optional["CostModel"],
        database: Optional["Database"],
        workload_key: Optional[Workload],
        use_llm: bool,
        llm_budget: int,
        llm_policy: Optional["LLMGuidancePolicy"] = None,
        llm_model_name: str = "",
    ):
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob  # unused by standard MCTS
        self.genetic_max_fail_count = genetic_max_fail_count
        self.num_empty_iters_before_early_stop = num_empty_iters_before_early_stop
        self.max_stale_iters = max_stale_iters
        self.diversity_epsilon = diversity_epsilon
        self.max_stale_diversity_iters = max_stale_diversity_iters
        self.trace_commit = trace_commit
        self.verbose = verbose

        self.mcts_ucb_constant = mcts_ucb_constant
        self.mcts_max_depth = mcts_max_depth
        self.mcts_num_threads = mcts_num_threads
        self.mcts_num_rollouts_per_expansion = mcts_num_rollouts_per_expansion

        self._postprocs = postprocs
        self._mutator_probs = mutator_probs
        self._ctx = context
        self._cost_model = cost_model
        self._database = database
        self._workload_key = workload_key

        self._workload_cache: Dict[int, Workload] = {}
        self._mutator_failure_count: Dict[object, int] = {"total": 0}
        self._search_state: Optional["MCTSTuningState"] = None

        self.use_llm = use_llm
        self.llm_budget = llm_budget
        self.llm_policy = llm_policy
        self.llm_model_name = llm_model_name

    def attach_search_state(self, search_state: "MCTSTuningState") -> None:
        self._search_state = search_state

    def explore(
        self,
        mcts_root: MCTSNode,
        population: List[Tuple[tvm.tir.Schedule, bool]],
        dynamic_pop_size: int,
        rand_state: int,
    ) -> List[Tuple[tvm.tir.Schedule, bool]]:
        logger.warning(
            "[DEBUG] explore() called with dynamic_pop_size=%d, genetic_num_iters=%d",
            dynamic_pop_size,
            self.genetic_num_iters
        )
        if not mcts_root or not mcts_root.children:
            logger.warning("explore(): Root is empty or has no children. Returning existing population.")
            return population
        total_expansions = 0
        for gen_iter in range(self.genetic_num_iters):
            new_children_count = 0
            fail_count = 0
            logger.warning("explore(): Starting generation %d ...", gen_iter)
            while new_children_count < dynamic_pop_size:
                # 1) UCB selection
                leaf = self._select(mcts_root)
                if leaf is None:
                    # means we couldn't pick a leaf at all
                    logger.warning(
                        "explore(): MCTS: Leaf is None in selection => break expansions."
                    )
                    break
                logger.warning(
                    "explore(): [gen=%d] Selected leaf node at depth=%d with %d children",
                    gen_iter, leaf.depth, len(leaf.children)
                )
                new_node = self._expand(leaf, rand_state)
                if new_node is None:
                    fail_count += 1
                    logger.warning(
                        "explore(): Failed to expand leaf at depth=%d (fail_count=%d)",
                        leaf.depth, fail_count
                    )
                    if fail_count >= self.genetic_max_fail_count:
                        logger.warning(
                            "explore(): Too many expansion failures => break expansions."
                        )
                        break
                    continue
                logger.warning(
                    "explore(): Successfully expanded leaf at depth=%d => new node at depth=%d; "
                    "now leaf has %d children total",
                    leaf.depth, new_node.depth, len(leaf.children)
                )
                value = self._simulate_node(new_node, rand_state)
                logger.warning(
                    "explore(): Simulation done for new node at depth=%d => value=%.4f",
                    new_node.depth, value
                )
                self._backprop(new_node, value)
                logger.warning(
                    "explore(): Backprop done => node visits=%d, total_value=%.4f",
                    new_node.visits, new_node.total_value
                )
                new_children_count += 1
                total_expansions += 1
            logger.warning(
                "explore(): [gen=%d] expansions=%d, fail_count=%d so far in this generation",
                gen_iter, new_children_count, fail_count
            )
        logger.warning(
            "explore(): All expansions complete => total_expansions=%d across %d generations.",
            total_expansions, self.genetic_num_iters
        )
        all_nodes = self._gather_tree_schedules(mcts_root)
        logger.warning(
            "explore(): Gathered %d total nodes (with schedules) from the MCTS tree.",
            len(all_nodes)
        )
        new_population = []
        for node in all_nodes:
            if node.schedule is not None:
                wl = self._commit_workload_cached(node.schedule)
                measured_flag = (wl is not None) and (wl in self._measured_workloads)
                new_population.append((node.schedule, measured_flag))
        logger.warning(
            "explore(): Returning a new population of size=%d (some may be measured).",
            len(new_population)
        )
        return new_population


    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Select a node for expansion or simulation using UCB-based traversal.
        """
        current = node
        while True:
            if current.parent is None: 
                unvisited = [c for c in current.children if c.visits == 0]
                if unvisited:
                    current = unvisited[0]
                else:
                    current = self._select_by_ucb(current)
                continue    
            if current.visits == 0:
                return current
            if len(current.children) < 2:
                return current
            unvisited_children = [child for child in current.children if child.visits == 0]
            if unvisited_children:
                return unvisited_children[0]
            next_child = self._select_by_ucb(current)
            if next_child is None:
                return None
            current = next_child

    def _select_by_ucb(self, node: MCTSNode) -> Optional[MCTSNode]:
        best_child = None
        best_score = -float("inf")
        c = self.mcts_ucb_constant
        for ch in node.children:
            if ch.visits == 0:
                return ch
            exploit = ch.total_value / ch.visits
            explore = math.sqrt(max(1e-12, math.log(node.visits) / ch.visits))
            score = exploit + c * explore
            if score > best_score:
                best_score = score
                best_child = ch
        return best_child

    def _expand(self, leaf: MCTSNode, rand_state: int) -> Optional[MCTSNode]:
        """
        Expand a leaf node by applying multiple random mutations to its schedule.
        """
        if len(leaf.children) >= 2:
            return None
        if self.mcts_max_depth is not None and leaf.depth >= self.mcts_max_depth:
            return None
        if leaf.schedule is None:
            return None
        can_use_llm = (
            self.use_llm
            and self.llm_policy is not None
            and self.llm_budget > 0
            and (2 <= leaf.depth)
        )
        if not can_use_llm:
            logger.warning("Not using LLM (either disabled, no budget, or depth/children constraints). Using random mutator.")
            new_sch = self._try_mcts_mutation(leaf.schedule, rand_state)
            if not new_sch:
                logger.warning("At line 625.")
                return None
            child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
            leaf.children.append(child)
            return child
        else:
            logger.warning("LLM usage is enabled. Gathering historical info for leaf, parent, and grandparent schedules.")
            new_sch = None
            historical_perf_parts = []
            try:
                leaf_score_list = self._predict_normalized_score([leaf.schedule])
                leaf_score = leaf_score_list[0] if leaf_score_list else 0.0
                try:
                    leaf_mod_str = leaf.schedule.mod.script()
                except Exception:
                    leaf_mod_str = "<failed to script IR>"
                leaf_trace_str = str(leaf.schedule.trace)
                historical_perf_parts.append(
                    "Current Schedule:\n"
                    f"Current Schedule's IR:\n{leaf_mod_str}\n\n"
                    f"Current Schedule's Trace:\n{leaf_trace_str}\n\n"
                    f"Current Schedule's Predicted Score by TVM's default cost model XGBoost: {leaf_score}\n"
                )
    
                parent_node = leaf.parent
                if parent_node and parent_node.schedule is not None:
                    p1_sch = parent_node.schedule
                    scores_p1 = self._predict_normalized_score([p1_sch])
                    score_p1 = scores_p1[0] if scores_p1 else 0.0
                    try:
                        p1_mod_str = p1_sch.mod.script()
                    except Exception:
                        p1_mod_str = "<failed to script IR>"
                    p1_trace_str = str(p1_sch.trace)
                    historical_perf_parts.append(
                        "Immediate Parent Schedule:\n"
                        f"Immediate Parent's IR:\n{p1_mod_str}\n\n"
                        f"Immediate Parent's Trace:\n{p1_trace_str}\n\n"
                        f"Immediate Parent's Predicted Score by TVM's default cost model XGBoost: {score_p1}\n"
                    )
    
                    grandparent_node = parent_node.parent
                    if grandparent_node and grandparent_node.schedule is not None:
                        p2_sch = grandparent_node.schedule
                        scores_p2 = self._predict_normalized_score([p2_sch])
                        score_p2 = scores_p2[0] if scores_p2 else 0.0
                        try:
                            p2_mod_str = p2_sch.mod.script()
                        except Exception:
                            p2_mod_str = "<failed to script IR>"
                        p2_trace_str = str(p2_sch.trace)
                        historical_perf_parts.append(
                            "Grandparent Schedule:\n"
                            f"Grandparent's IR:\n{p2_mod_str}\n\n"
                            f"Grandparent's Trace:\n{p2_trace_str}\n\n"
                            f"Grandparent's Predicted Score by TVM's default cost model XGBoost: {score_p2}\n"
                        )
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning("Failed to gather historical info for Leaf/Parent/Grandparent: %s", str(e))
    
            historical_perf = "\n\n".join(historical_perf_parts) if historical_perf_parts else None
    
            logger.warning("Invoking LLM policy to pick a sequence of mutators.")
            possible_mutator_names = [str(m) for m in self._mutator_probs.keys()]
            mutator_probs_dict = {str(mut): prob for mut, prob in self._mutator_probs.items()}
            chosen_mutator_names = self.llm_policy.pick_mutators(
                mod=leaf.schedule.mod,
                available_mutators=possible_mutator_names,
                historical_perf=historical_perf,
                available_mutator_probs=mutator_probs_dict,
            )
            if chosen_mutator_names is not None and len(chosen_mutator_names) > 0:
                logger.warning("LLM returned mutator names: '%s'", chosen_mutator_names)
                temp_sch = leaf.schedule
                for name in chosen_mutator_names:
                    chosen_mutator = None
                    for mut, _prob in self._mutator_probs.items():
                        if str(mut) == name:
                            chosen_mutator = mut
                            break
                    if chosen_mutator is None:
                        logger.warning(
                            "LLM mutator name '%s' did not match any known mutator. Fallback to random for this step.",
                            name
                        )
                        chosen_mutator = self._pick_random_mutator(rand_state)
                    maybe_new = self._apply_mutator_with_retry(temp_sch, chosen_mutator, rand_state)
                    if maybe_new is None:
                        logger.warning("Failed applying mutator '%s'. Will not continue sequence.", name)
                        break
                    temp_sch = maybe_new

                new_sch = temp_sch
                self.llm_budget -= 1
                logger.warning("LLM budget decremented. Remaining: %d", self.llm_budget)
            else:
                logger.warning("LLM did not produce a valid mutator list => fallback to a random single mutation.")
                new_sch = self._try_mcts_mutation(leaf.schedule, rand_state)
            if not new_sch:
                logger.warning("Failed to create a new schedule from chosen mutators. Expansion returning None.")
                return None

            child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
            leaf.children.append(child)
            logger.warning(
                "Successfully expanded leaf using %s approach. New child node at depth %d.",
                "LLM-based" if chosen_mutator_names else "random",
                child.depth
            )
            return child


    def _simulate_node(self, node: MCTSNode, rand_state: int) -> float:
        """
        Run rollout/simulation from a newly expanded node
        """
        if (self.mcts_num_rollouts_per_expansion <= 1) and (self.mcts_num_threads <= 1):
            return self._rollout(node.schedule, node.depth, rand_state)

        results = []
        if (self.mcts_num_threads > 1) and (self.mcts_num_rollouts_per_expansion > 1):
            with ThreadPoolExecutor(max_workers=self.mcts_num_threads) as executor:
                futures = [
                    executor.submit(self._rollout, node.schedule, node.depth, rand_state)
                    for _ in range(self.mcts_num_rollouts_per_expansion)
                ]
                for f in as_completed(futures):
                    results.append(f.result())
        else:
            for _ in range(self.mcts_num_rollouts_per_expansion):
                results.append(self._rollout(node.schedule, node.depth, rand_state))
        if results:
            return sum(results) / len(results)
        return 0.0

    def _backprop(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate the 'value' up the tree, incrementing visits and
        adding 'value' to total_value.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _rollout(self, schedule: Schedule, depth: int, rand_state: int) -> float:
        new_sch = self._replay_schedule(schedule.trace, rand_state)
        if new_sch is None:
            return 0.0
        cur_depth = depth
        while (self.mcts_max_depth is None) or (cur_depth < self.mcts_max_depth):
            cur_depth += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                logger.warning("[_rollout] No mutator found (mut is None). Breaking from rollout loop.")
                break
            try:
                mutated_trace = mut.apply(new_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                mutated_trace = None
            if mutated_trace is None:
                logger.warning("[_rollout] mutated_trace is None after apply(). Stopping mutations.")
                break
            maybe_new = self._replay_schedule(mutated_trace, rand_state)
            if maybe_new is None:
                logger.warning("[_rollout] Replaying the mutated trace returned None. Stopping mutations.")
                break
            new_sch = maybe_new
            # logger.warning(
            #     f"[_rollout] Successfully replayed mutated trace. Now at rollout depth={cur_depth}."
            # )
        if not self._cost_model or not self._database:
            logger.warning(
                f"[_rollout] No cost_model or no database found. Returning random fallback score."
            )
            return random.random()  # fallback random
        arg_info = ArgInfo.from_entry_func(new_sch.mod, remove_preproc=True)
        candidate = MeasureCandidate(new_sch, arg_info)
        preds = self._cost_model.predict(self._ctx, [candidate])
        if preds:
            logger.warning(
                f"[_rollout] Final cost-model prediction. Returning this from rollout."
            )
            return max(0.0, preds[0])
        return 0.0

    def gather_unmeasured_leaves(self, node: MCTSNode) -> List[MCTSNode]:
        stack = [node]
        leaves = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None and not nd.children:
                wl = None
                wl = self._commit_workload_cached(nd.schedule)
                if wl not in self._measured_workloads:
                    leaves.append(nd)
            else:
                stack.extend(nd.children)
        return leaves

    def pick_unmeasured_best_leaves(self, root: MCTSNode, batch_size: int) -> List[Schedule]:
        leaves = self.gather_unmeasured_leaves(root)
        if not leaves:
            return []
        scored = []
        for nd in leaves:
            q_val = (nd.total_value / nd.visits) if nd.visits > 0 else 0.0
            scored.append((nd, q_val))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored[:batch_size]
        return [node.schedule for (node, _) in top_nodes]

    def _gather_tree_schedules(self, root: MCTSNode) -> List[MCTSNode]:
        stack = [root]
        out_nodes = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None:
                out_nodes.append(nd)
            stack.extend(nd.children)
        return out_nodes


    def _replay_schedule(self, trace: Optional[Trace], rand_state: int) -> Optional[Schedule]:
        if not self._ctx or not self._ctx.mod:
            return None
        mod = self._ctx.mod
        if trace is None:
            try:
                sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            except (InvalidScheduleError, tvm.TVMError):
                return None
            sch.enter_postproc()
            if not self._apply_postprocs(sch):
                return None
            return sch
        try:
            sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None
        sch.enter_postproc()
        if not self._apply_postprocs(sch):
            return None
        return sch

    def _apply_postprocs(self, sch: Schedule) -> bool:
        if not self._postprocs:
            return True
        ffi_postproc = getattr(_ffi_api, "SearchStrategyApplyPostprocs", None)
        if ffi_postproc is not None:
            try:
                return bool(ffi_postproc(sch, self._postprocs))
            except Exception:
                pass
        for proc in self._postprocs:
            try:
                if not proc.apply(sch):
                    return False
            except (InvalidScheduleError, tvm.TVMError):
                return False
        return True


    def _pick_random_mutator(self, rand_state: int) -> Optional[Mutator]:
        if not self._mutator_probs:
            return None
        total_p = sum(self._mutator_probs.values())
        r = random.random() * total_p
        s = 0.0
        for mut, p in self._mutator_probs.items():
            s += p
            if r <= s:
                return mut
        return list(self._mutator_probs.keys())[0]

    def _try_mcts_mutation(self, parent_sch: Schedule, rand_state: int) -> Optional[Schedule]:
        attempts = 0
        while attempts <= self.genetic_max_fail_count:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                child_sch = self._replay_schedule(parent_sch.trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    # wl = self._database.commit_workload(child_sch.mod)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
                continue
            try:
                new_trace = mut.apply(parent_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                new_trace = None
            if new_trace is None:
                self._mutator_failure_count[mut] = self._mutator_failure_count.get(mut, 0) + 1
            else:
                child_sch = self._replay_schedule(new_trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    # wl = self._database.commit_workload(child_sch.mod)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
        return None

    def _apply_mutator_with_retry(
        self,
        parent_sch: tvm.tir.Schedule,
        chosen_mutator: Mutator,
        rand_state: int
    ) -> Optional[tvm.tir.Schedule]:
        attempts = 0
        while attempts <= self.genetic_max_fail_count:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            try:
                new_trace = chosen_mutator.apply(parent_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                new_trace = None
            if new_trace is None:
                self._mutator_failure_count[chosen_mutator] = (
                    self._mutator_failure_count.get(chosen_mutator, 0) + 1
                )
            else:
                child_sch = self._replay_schedule(new_trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._commit_workload_cached(child_sch)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
        return None
    
    def _get_dynamic_ucb_constant(self) -> float:
        c0 = self.mcts_ucb_constant
        alpha = 0.99
        scaling = 150.0
        current_trials = 0
        if self._search_state is not None:
            current_trials = self._search_state.trial_count
        exponent = float(current_trials) / scaling
        c_dynamic = c0 * (alpha ** exponent)
        if self.verbose >= 2:
            logger.debug(
                f"[_get_dynamic_ucb_constant] trial_count={current_trials}, c_dynamic={c_dynamic:.4f}"
            )
        return c_dynamic


    def _commit_workload_cached(self, sch: Schedule) -> Optional[Workload]:
        if self._database is None:
            return None
        wl = getattr(sch, "_cached_wl", None)
        if wl is not None:
            return wl
        shash = tvm.ir.structural_hash(sch.mod)
        wl = self._workload_cache.get(shash)
        if wl is None:
            wl = self._database.commit_workload(sch.mod)
            self._workload_cache[shash] = wl
        sch._cached_wl = wl
        return wl

    def _predict_normalized_score(self, schedules: List[Schedule]) -> List[float]:
        if not schedules or not self._cost_model or not self._database:
            return [0.0] * len(schedules)
        cands = []
        for sch in schedules:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            cands.append(MeasureCandidate(sch, arg_info))
        scores = self._cost_model.predict(self._ctx, cands)
        return [max(0.0, sc) for sc in scores]

    @property
    def _measured_workloads(self) -> Set[Workload]:
        """
        The set of workload keys that have been actually measured on hardware.
        """
        if self._search_state is not None:
            return self._search_state.measured_workloads
        return set()

    @property
    def _seen_workloads(self) -> Set[Workload]:
        """
        The set of workload keys encountered in generated schedules.
        """
        if self._search_state is not None:
            return self._search_state.seen_workloads
        return set()


class MCTSTuningState:
    """
    MCTSTuningState tracks the MCTS root, population, number of trials used,
    best score, etc. The MCTSTuner performs expansions and rollouts; 
    MCTSTuningState decides how to handle each iteration.
    """

    def __init__(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"],
        cost_model: Optional["CostModel"],
        context: "TuneContext",
        tuner: MCTSTuner,
    ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_spaces = design_spaces
        self.database = database
        self.cost_model = cost_model
        self.context = context
        self.tuner = tuner
        self.tuner.attach_search_state(self)

        self.mod = context.mod
        self.workload_key = None
        if self.database and self.mod is not None:
            self.workload_key = self.database.commit_workload(self.mod)
            self.tuner._workload_key = self.workload_key

        self.trial_count = 0
        self.num_empty_iters = 0
        self.used_init_population = False
        self.population: List[Tuple[Schedule, bool]] = []
        self.mcts_root: Optional[MCTSNode] = None

        self.measured_workloads: Set[Workload] = set()
        self.seen_workloads: Set[Workload] = set()

        self.best_score_so_far = -float("inf")
        self.stale_iter_count = 0
        self.stale_diversity_count = 0
        self.diversity_history: List[float] = []
        self.score_history: List[float] = []
        self.dynamic_pop_size = self.tuner.population_size

        rs = context.rand_state
        self.rand_state = rs if rs is not None else 1
        if self.rand_state == 0:
            self.rand_state = 1

    def reset(self) -> None:
        """
        Called from MCTSSearch.post_tuning().
        """

    
    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        if self.tuner.verbose >= 1:
            logger.warning(
                "[DEBUG] Enter generate_measure_candidates: trial_count=%d, max_trials=%d",
                self.trial_count, self.max_trials
            )
        if self.trial_count >= self.max_trials:
            return None
        remaining = self.max_trials - self.trial_count
        batch_size = min(remaining, self.num_trials_per_iter)
        if batch_size <= 0:
            return None
        if not self.used_init_population:
            init_pop = self._init_population()
            if not init_pop:
                return None
            self.mcts_root = MCTSNode(schedule=None, parent=None, depth=0)
            for (sch, is_measured) in init_pop:
                child = MCTSNode(schedule=sch, parent=self.mcts_root, depth=1)
                self.mcts_root.children.append(child)
            self.population = init_pop
            self.used_init_population = True
            if self.tuner.verbose >= 1:
                logger.warning(
                    "generate_measure_candidates: MCTS: Initialized root with %d child schedules.",
                    len(init_pop)
                )

        self.population = self.tuner.explore(
            mcts_root=self.mcts_root,
            population=self.population,
            dynamic_pop_size=self.dynamic_pop_size,
            rand_state=self.rand_state,
        )
        if not self.population:
            self.num_empty_iters += 1
            logger.warning(
                "generate_measure_candidates: MCTS: explore() returned empty => empty iters=%d",
                self.num_empty_iters
            )
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                if self.tuner.verbose >= 1:
                    logger.warning("generate_measure_candidates: MCTS: Stopping early => repeated empty iters.")
                return None
            return None
        logger.warning(
            "generate_measure_candidates: MCTS: population size=%d before eps-greedy picking",
            len(self.population)
        )
        cands_sch = self._pick_unmeasured_eps_greedy(self.population, batch_size, self.rand_state)
        if not cands_sch:
            self.num_empty_iters += 1
            logger.warning(
                "generate_measure_candidates: MCTS: no unmeasured schedules => empty iters=%d",
                self.num_empty_iters
            )
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                if self.tuner.verbose >= 1:
                    logger.warning("generate_measure_candidates: stopping early => repeated empty iters.")
                return None
            return None
        logger.warning(
            "generate_measure_candidates: [DEBUG] Eps-greedy picked %d schedules for measurement (batch_size=%d).",
            len(cands_sch), batch_size
        )
        measure_cands: List[MeasureCandidate] = []
        for sch in cands_sch:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            measure_cands.append(MeasureCandidate(sch, arg_info))
        logger.warning(
                "generate_measure_candidates: [DEBUG] MCTS => returning %d cands; trial_count=%d, "
                "batch_size_requested=%d, used_init_population=%s",
                len(measure_cands),
                self.trial_count,
                batch_size,
                str(self.used_init_population),
            )
        return measure_cands


    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        if self.database is None:
            logger.warning("database is not defined, skipping MCTS measure update.")
            return

        num_measured_now = 0
        best_run_sec = float("inf")
        for cand, res in zip(measure_candidates, results):
            sch = cand.sch
            mod = sch.mod
            wl = self.database.commit_workload(mod)
            if res.run_secs and all(t >= 0 for t in res.run_secs):
                run_sec = sum(res.run_secs) / len(res.run_secs)
                if run_sec < best_run_sec:
                    best_run_sec = run_sec
                self.measured_workloads.add(wl)
                self._mark_schedule_measured(sch)
                num_measured_now += 1
        self.trial_count += num_measured_now
        if best_run_sec < float("inf"):
            new_score = 1.0 / best_run_sec
            self.score_history.append(new_score)
            if new_score > self.best_score_so_far + 1e-12:
                self.best_score_so_far = new_score
                self.stale_iter_count = 0
            else:
                self.stale_iter_count += 1
                if self.stale_iter_count >= self.tuner.max_stale_iters and self.tuner.verbose >= 1:
                    logger.warning(
                        "notifu_runner_results: MCTS: No improvement => stopping early (stale_iter=%d).",
                        self.stale_iter_count
                    )
        else:
            self.score_history.append(0.0)

        # # Check population diversity
        # if self.population:
        #     pop_scores = self._predict_population_scores(self.population)
        #     diversity = self._check_population_diversity(pop_scores)
        #     if diversity < self.tuner.diversity_epsilon:
        #         self.stale_diversity_count += 1
        #         if self.tuner.verbose >= 1:
        #             logger.info(
        #                 "MCTS: Pop diversity=%.6f < threshold => stale_diversity_count=%d",
        #                 diversity, self.stale_diversity_count
        #             )
        #         if self.stale_diversity_count >= self.tuner.max_stale_diversity_iters:
        #             if self.tuner.verbose >= 1:
        #                 logger.info("MCTS: Population too homogeneous => stopping early.")
        #     else:
        #         self.stale_diversity_count = 0

        # Adaptive population resizing example
        #if self.population:
        #    if self.stale_diversity_count > 0:
        #        old_size = self.dynamic_pop_size
        #        self.dynamic_pop_size = max(10, int(self.dynamic_pop_size * 0.9))
        #        if self.tuner.verbose >= 1:
        #            logger.info(
        #                "MCTS: Adaptive pop resize: %d -> %d",
        #                old_size, self.dynamic_pop_size
        #            )
        #    else:
        #        self.dynamic_pop_size = min(
        #            self.tuner.population_size,
        #            self.dynamic_pop_size + 5
        #        )

        if self.tuner.verbose >= 1:
            logger.warning(
                "MCTS: notify_runner_results => measured=%d, total=%d, stale_iter=%d, div_stale=%d",
                num_measured_now, self.trial_count, self.stale_iter_count, self.stale_diversity_count
            )

    def _init_population(self) -> List[Tuple[Schedule, bool]]:
        """
        Combine schedules from DB (measured) and random design-space samples (unmeasured).
        """
        num_measured_wanted = int(self.tuner.population_size * self.tuner.init_measured_ratio)
        measured_from_db = self._pick_best_from_database(num_measured_wanted)
        need_rand = max(
            self.tuner.population_size - len(measured_from_db),
            self.tuner.init_min_unmeasured
        )
        unmeasured_rand = self._sample_init_population(need_rand)
        logger.warning(
            "[MCTS init_pop] from DB: %d, from random: %d, population_size=%d, init_min_unmeasured=%d",
            len(measured_from_db),
            len(unmeasured_rand),
            self.tuner.population_size,
            self.tuner.init_min_unmeasured
        )
        combined = [(sch, True) for sch in measured_from_db] + \
                   [(sch, False) for sch in unmeasured_rand]
        random.shuffle(combined)
        if len(combined) > self.tuner.population_size:
            combined = combined[: self.tuner.population_size]
        for (sch, measured_flag) in combined:
            wl = self.tuner._commit_workload_cached(sch)
            self.seen_workloads.add(wl)
            if measured_flag:
                self.measured_workloads.add(wl)
        return combined

    def _pick_best_from_database(self, num: int) -> List[Schedule]:
        if num <= 0 or not self.database:
            return []
        out = []
        top_records = self.database.get_top_k(self.workload_key, num)
        for rec in top_records:
            sch = self._replay_schedule(rec.trace)
            if sch is not None:
                wl = self.tuner._commit_workload_cached(sch)
                # wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
        return out

    def _replay_schedule(self, trace: Optional[Trace]) -> Optional[Schedule]:
        if not trace or not self.context or not self.context.mod:
            return None
        mod = self.context.mod
        try:
            sch = Schedule(mod, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None
        sch.enter_postproc()
        for proc in self.tuner._postprocs:
            try:
                if not proc.apply(sch):
                    return None
            except (InvalidScheduleError, tvm.TVMError):
                return None
        return sch

    def _sample_init_population(self, num: int) -> List[Schedule]:
        out = []
        fails = 0
        n_spaces = len(self.design_spaces)
        while len(out) < num and fails < self.tuner.max_fail_count:
            idx = random.randint(0, n_spaces - 1)
            base_sch = self.design_spaces[idx]
            sch = self._replay_schedule(base_sch.trace)
            if sch is not None:
                wl = self.tuner._commit_workload_cached(sch)
                # wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
                    self.seen_workloads.add(wl)
                else:
                    fails += 1
            else:
                fails += 1
        return out
    
    def _pick_unmeasured_eps_greedy(
        self,
        schedules_with_flags: List[Tuple[tvm.tir.Schedule, bool]],
        total_needed: int,
        rand_state: int
    ) -> List[tvm.tir.Schedule]:
        logger.warning(
            "[DEBUG] _pick_unmeasured_eps_greedy called with total_needed=%d, eps_greedy=%.3f",
            total_needed, 0.05
        )
        unmeasured = []
        for (sch, measured_flag) in schedules_with_flags:
            if not measured_flag:
                unmeasured.append(sch)
        logger.warning("[DEBUG] Found %d unmeasured schedules.", len(unmeasured))
        if not unmeasured:
            return []
        preds = self.tuner._predict_normalized_score(unmeasured)
        logger.warning("[DEBUG] Computed cost-model predictions for %d unmeasured schedules.", len(preds))
        scored = list(zip(unmeasured, preds))
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.warning(
            "[DEBUG] Top schedule after sorting has predicted score=%.4f if the list is non-empty.",
            scored[0][1] if scored else -1.0
        )
        n_total = min(total_needed, len(scored))
        n_rand = int(round(n_total * 0.05))
        n_top = n_total - n_rand
        logger.warning(
            "[DEBUG] Eps-greedy selection: total_needed=%d => n_top=%d, n_rand=%d",
            n_total, n_top, n_rand
        )
        top_part = scored[:n_top]
        leftover = scored[n_top:]
        random_schedules = []
        if leftover and n_rand > 0:
            random.seed(rand_state)
            n_rand = min(n_rand, len(leftover))
            random_part = random.sample(leftover, n_rand)
            random_schedules = [sch for (sch, _) in random_part]
        top_schedules = [sch for (sch, _) in top_part]
        combined = top_schedules + random_schedules
        logger.warning(
            "[DEBUG] _pick_unmeasured_eps_greedy => returning %d schedules => %d top + %d random",
            len(combined), len(top_schedules), len(random_schedules)
        )
        return combined

    def _mark_schedule_measured(self, sch: Schedule):
        wl = self.database.commit_workload(sch.mod)
        self.measured_workloads.add(wl)
        for i, (pop_sch, was_measured) in enumerate(self.population):
            if pop_sch == sch and not was_measured:
                self.population[i] = (pop_sch, True)


    def _predict_population_scores(self, pop: List[Tuple[Schedule, bool]]) -> List[float]:
        schs = [p[0] for p in pop]
        if not schs:
            return []
        return self.tuner._predict_normalized_score(schs)

    def _check_population_diversity(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        mean_val = sum(scores) / len(scores)
        var = sum((s - mean_val) ** 2 for s in scores) / len(scores)
        cur_div = math.sqrt(var)
        self.diversity_history.append(cur_div)
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        avg_div = sum(self.diversity_history) / len(self.diversity_history)
        self.tuner.diversity_epsilon = 0.5 * avg_div
        return cur_div



@derived_object
class MCTSSearchPyFull(PySearchStrategy):
    def __init__(
        self,
        population_size: int = 3,
        init_measured_ratio: float = 0.0,
        init_min_unmeasured: int = 3,
        max_fail_count: int = 20,
        genetic_num_iters: int = 2,
        genetic_mutate_prob: float = 0.85,
        genetic_max_fail_count: int = 2,
        num_empty_iters_before_early_stop: int = 10,
        max_stale_iters: int = 60,
        diversity_epsilon: float = 1e-6,
        max_stale_diversity_iters: int = 30,
        trace_commit: bool = True,
        verbose: int = 2,
        # MCTS-specific:
        mcts_ucb_constant: float = 1.41,
        mcts_max_depth: Optional[int] = 500,
        mcts_num_threads: int = 1,
        mcts_num_rollouts_per_expansion: int = 1,
        use_llm: bool = False,
        llm_budget: int = 1,
        llm_model_name: str = "",
    ) -> None:
        super().__init__()
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob
        self.genetic_max_fail_count = genetic_max_fail_count
        self.num_empty_iters_before_early_stop = num_empty_iters_before_early_stop
        self.max_stale_iters = max_stale_iters
        self.diversity_epsilon = diversity_epsilon
        self.max_stale_diversity_iters = max_stale_diversity_iters
        self.trace_commit = trace_commit
        self.verbose = verbose

        self.mcts_ucb_constant = mcts_ucb_constant
        self.mcts_max_depth = mcts_max_depth
        self.mcts_num_threads = mcts_num_threads
        self.mcts_num_rollouts_per_expansion = mcts_num_rollouts_per_expansion
        self.use_llm = use_llm
        self.llm_budget = llm_budget
        self._ctx: Optional["TuneContext"] = None
        self._postprocs: List[Postproc] = []
        self._mutator_probs: Dict[Mutator, float] = {}
        self.state: Optional[MCTSTuningState] = None

        self.llm_model_name = llm_model_name

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        self._ctx = context
        if context.space_generator is None:
            raise ValueError("TuneContext.space_generator must be defined.")
        if context.target is None:
            raise ValueError("TuneContext.target must be defined.")
        sg = context.space_generator
        self._postprocs = list(sg.postprocs) if sg.postprocs else []
        user_probs = sg.mutator_probs or {}
        for mut, prob_f in user_probs.items():
            p = float(prob_f.value)
            if p > 1e-12:
                self._mutator_probs[mut] = self._mutator_probs.get(mut, 0.0) + p
        target_kind = str(context.target.kind.name)
        if not self._mutator_probs:
            try:
                default_muts = Mutator.create(target_kind)
            except:
                default_muts = Mutator.create("llvm")
            if isinstance(default_muts, dict):
                for m, p2 in default_muts.items():
                    self._mutator_probs[m] = float(p2)
            elif isinstance(default_muts, list) and len(default_muts) > 0:
                p2 = 1.0 / len(default_muts)
                for m in default_muts:
                    self._mutator_probs[m] = p2
        total_p = sum(self._mutator_probs.values())
        if total_p > 1e-12:
            for k in self._mutator_probs:
                self._mutator_probs[k] /= total_p
        if self.verbose >= 1:
            logger.warning(
                "_initialize_with_tune_context: MCTSSearch: Using target=%s, found #mutators=%d, rand_state=%s",
                target_kind, len(self._mutator_probs), str(context.rand_state)
            )

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"],
        cost_model: Optional["CostModel"],
    ) -> None:
        if self.state is not None:
            raise ValueError("MCTSSearch.pre_tuning called without post_tuning after previous run")

        tuner = MCTSTuner(
            population_size=self.population_size,
            init_measured_ratio=self.init_measured_ratio,
            init_min_unmeasured=self.init_min_unmeasured,
            max_fail_count=self.max_fail_count,
            genetic_num_iters=self.genetic_num_iters,
            genetic_mutate_prob=self.genetic_mutate_prob,
            genetic_max_fail_count=self.genetic_max_fail_count,
            num_empty_iters_before_early_stop=self.num_empty_iters_before_early_stop,
            max_stale_iters=self.max_stale_iters,
            diversity_epsilon=self.diversity_epsilon,
            max_stale_diversity_iters=self.max_stale_diversity_iters,
            trace_commit=self.trace_commit,
            verbose=self.verbose,
            mcts_ucb_constant=self.mcts_ucb_constant,
            mcts_max_depth=self.mcts_max_depth,
            mcts_num_threads=self.mcts_num_threads,
            mcts_num_rollouts_per_expansion=self.mcts_num_rollouts_per_expansion,
            postprocs=self._postprocs,
            mutator_probs=self._mutator_probs,
            context=self._ctx,
            cost_model=cost_model,
            database=database,
            workload_key=None,
            use_llm=self.use_llm,
            llm_budget=self.llm_budget,
            llm_policy=LLMGuidancePolicy(
            model_name=self.llm_model_name,
            verbose=True),
        )
        self.state = MCTSTuningState(
            max_trials=max_trials,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=design_spaces,
            database=database,
            cost_model=cost_model,
            context=self._ctx,
            tuner=tuner,
        )
        if self.verbose >= 1:
            logger.warning(
                "MCTSSearch.pre_tuning => max_trials=%d, num_per_iter=%d, #design_spaces=%d",
                max_trials, num_trials_per_iter, len(design_spaces)
            )

    def post_tuning(self) -> None:
        if self.state:
            self.state.reset()
            self.state = None
        if self.verbose >= 1:
            logger.warning("MCTSSearch: Tuning finished in post_tuning().")

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        if not self.state:
            logger.warning("MCTSSearch.generate_measure_candidates called before pre_tuning.")
            return None
        return self.state.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        if self.state:
            self.state.notify_runner_results(measure_candidates, results)

    def clone(self) -> "MCTSSearchPyFull":
        return MCTSSearchPyFull(
            population_size=self.population_size,
            init_measured_ratio=self.init_measured_ratio,
            init_min_unmeasured=self.init_min_unmeasured,
            max_fail_count=self.max_fail_count,
            genetic_num_iters=self.genetic_num_iters,
            genetic_mutate_prob=self.genetic_mutate_prob,
            genetic_max_fail_count=self.genetic_max_fail_count,
            num_empty_iters_before_early_stop=self.num_empty_iters_before_early_stop,
            max_stale_iters=self.max_stale_iters,
            diversity_epsilon=self.diversity_epsilon,
            max_stale_diversity_iters=self.max_stale_diversity_iters,
            trace_commit=self.trace_commit,
            verbose=self.verbose,
            mcts_ucb_constant=self.mcts_ucb_constant,
            mcts_max_depth=self.mcts_max_depth,
            mcts_num_threads=self.mcts_num_threads,
            mcts_num_rollouts_per_expansion=self.mcts_num_rollouts_per_expansion,
            use_llm=self.use_llm,
            llm_budget=self.llm_budget,
            llm_model_name=self.llm_model_name,
        )


