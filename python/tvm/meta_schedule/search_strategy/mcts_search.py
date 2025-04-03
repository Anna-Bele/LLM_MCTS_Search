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


# -----------------------------------------------------------------------------
# Helper: A sized min-heap for storing top-K schedules by predicted score
# -----------------------------------------------------------------------------
class _SizedMinHeap:
    """
    Maintains top size_limit items by real score, storing an internal tuple:
      (neg_score, push_counter, schedule, measured_flag)

    The 'push_counter' is a strictly increasing tie-breaker so that Python's
    heapq never needs to compare two Schedule objects directly.

    Used below for population ranking based on cost-model predictions.
    """

    def __init__(self, size_limit: int):
        self._size_limit = size_limit
        self._heap = []
        self._push_counter = 0  # strictly increasing for tie-breaking

    def push(self, sch: Schedule, score: float, measured_flag: bool) -> None:
        """
        Push a schedule with a numeric 'score' into the min-heap. We keep only
        the top 'size_limit' schedules (by largest score).
        """
        neg_score = -score
        self._push_counter += 1
        item = (neg_score, self._push_counter, sch, measured_flag)
        if len(self._heap) < self._size_limit:
            heapq.heappush(self._heap, item)
        else:
            worst_neg, _, _, _ = self._heap[0]
            # If the new item is "better" => negative score is more negative => bigger real score
            if neg_score > worst_neg:
                # means new item is worse; discard it
                return
            heapq.heapreplace(self._heap, item)

    def items_descending(self) -> List[Tuple[float, Schedule, bool]]:
        """
        Return all items in descending order by 'score' (i.e., -neg_score).
        """
        items = []
        for (neg, _, sch, meas) in self._heap:
            score = -neg
            items.append((score, sch, meas))
        items.sort(key=lambda x: x[0], reverse=True)
        return items


# -----------------------------------------------------------------------------
# MCTS Node
# -----------------------------------------------------------------------------
class MCTSNode:
    """
    A node in the MCTS tree.

    Attributes
    ----------
    schedule : Optional[Schedule]
        The schedule at this node (None if root).

    parent : Optional[MCTSNode]
        The parent node in the MCTS tree.

    children : List[MCTSNode]
        Child nodes in the tree.

    visits : int
        Number of visits during selection/backprop.

    total_value : float
        Accumulated value from rollouts or expansions.

    depth : int
        The depth of this node in the tree (0 = root).
    """

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
        """
        Recursively clones this node (and its sub-tree).
        """
        new_node = MCTSNode(self.schedule, None, self.depth)
        new_node.visits = self.visits
        new_node.total_value = self.total_value
        for ch in self.children:
            child_copy = ch.clone_tree()
            child_copy.parent = new_node
            new_node.children.append(child_copy)
        return new_node


# -----------------------------------------------------------------------------
# MCTSTuner: Core MCTS logic (selection, expansion, simulation, backprop)
# -----------------------------------------------------------------------------
class MCTSTuner:
    """
    Implements the core Monte Carlo Tree Search routines:
      - UCB-based node selection
      - expansions
      - multiple rollouts (simulations)
      - backprop
      - cost model predictions
      - picking unmeasured leaves, ranking population
    """

    def __init__(
        self,
        population_size: int,
        init_measured_ratio: float,
        init_min_unmeasured: int,
        max_fail_count: int,
        genetic_num_iters: int,
        genetic_mutate_prob: float,  # Unused in MCTS, but kept for interface
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

        # references
        self._postprocs = postprocs
        self._mutator_probs = mutator_probs
        self._ctx = context
        self._cost_model = cost_model
        self._database = database
        self._workload_key = workload_key

        # track how many times a mutator fails
        self._mutator_failure_count: Dict[object, int] = {"total": 0}

        # Will be set by attach_search_state()
        self._search_state: Optional["MCTSTuningState"] = None

    def attach_search_state(self, search_state: "MCTSTuningState") -> None:
        """
        Provide the MCTSTuner a link back to MCTSTuningState so that it can
        read measured/unmeasured sets, etc.
        """
        self._search_state = search_state

    # -------------------------------------------------------------------------
    # Main exploration entrypoint
    # -------------------------------------------------------------------------
    def explore(
        self,
        mcts_root: MCTSNode,
        population: List[Tuple[Schedule, bool]],
        dynamic_pop_size: int,
        rand_state: int,
    ) -> List[Tuple[Schedule, bool]]:
        """
        Perform expansions (self.genetic_num_iters times) from mcts_root,
        gather all nodes, rank them with cost model => return new population
        of size dynamic_pop_size.
        """
        if not mcts_root or not mcts_root.children:
            # nothing to explore
            return population

        # expansions
        for _ in range(self.genetic_num_iters):
            leaf = self._select(mcts_root)
            if leaf is None:
                break
            new_node = self._expand(leaf, rand_state)
            if not new_node:
                continue
            value = self._simulate_node(new_node, rand_state)
            self._backprop(new_node, value)

        # gather all nodes => build new population
        all_nodes = self._gather_tree_schedules(mcts_root)
        pop_schedules = [node.schedule for node in all_nodes if node.schedule is not None]

        # rank by cost-model predictions
        heap = _SizedMinHeap(size_limit=dynamic_pop_size)
        preds = self._predict_normalized_score(pop_schedules)
        for sch, pred_score in zip(pop_schedules, preds):
            wl = None
            if self._database:
                wl = self._database.commit_workload(sch.mod)
            measured_flag = (wl is not None) and (wl in self._measured_workloads)
            heap.push(sch, pred_score, measured_flag)

        best_items = heap.items_descending()
        new_population = [(sch, meas) for (score, sch, meas) in best_items]
        return new_population

    # -------------------------------------------------------------------------
    # MCTS steps: selection, expansion, simulation, backprop
    # -------------------------------------------------------------------------
    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Select a leaf node by descending the tree using a UCB1 policy.
        """
        current = node
        while True:
            if not current.children:
                return current
            unvisited = [ch for ch in current.children if ch.visits == 0]
            if unvisited:
                return unvisited[0]
            next_child = self._select_by_ucb(current)
            if next_child is None:
                return None
            current = next_child

    def _select_by_ucb(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Choose child with the highest UCB1 score: Q + c*sqrt(log(N)/n).
        """
        best_child = None
        best_score = -float("inf")
        c = self.mcts_ucb_constant
        for ch in node.children:
            if ch.visits == 0:
                return ch
            exploit = ch.total_value / ch.visits
            # Avoid log(0)
            explore = math.sqrt(max(1e-12, math.log(node.visits) / ch.visits))
            score = exploit + c * explore
            if score > best_score:
                best_score = score
                best_child = ch
        return best_child

    def _expand(self, leaf: MCTSNode, rand_state: int) -> Optional[MCTSNode]:
        """
        Expand a leaf node by applying one random mutation to its schedule,
        if within max_depth.
        """
        if self.mcts_max_depth is not None and leaf.depth >= self.mcts_max_depth:
            return None
        if leaf.schedule is None:
            return None
        new_sch = self._try_mcts_mutation(leaf.schedule, rand_state)
        if not new_sch:
            return None
        child = MCTSNode(schedule=new_sch, parent=leaf, depth=leaf.depth + 1)
        leaf.children.append(child)
        return child

    def _simulate_node(self, node: MCTSNode, rand_state: int) -> float:
        """
        Run multiple rollouts from a newly expanded node:
         - if num_threads>1, run them concurrently
         - average the results
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

    # -------------------------------------------------------------------------
    # Simulation / rollout
    # -------------------------------------------------------------------------
    def _rollout(self, schedule: Schedule, depth: int, rand_state: int) -> float:
        """
        1) Replay the schedule from scratch,
        2) Repeatedly mutate it until hitting max depth or a mutation fails,
        3) Use cost model to predict a final score => return as the rollout's value.
        """
        # 1) replay the schedule from scratch
        new_sch = self._replay_schedule(schedule.trace, rand_state)
        if new_sch is None:
            return 0.0

        # 2) mutate until we hit max depth or fail
        cur_depth = depth
        while (self.mcts_max_depth is None) or (cur_depth < self.mcts_max_depth):
            cur_depth += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                break
            try:
                mutated_trace = mut.apply(new_sch.trace)
            except (InvalidScheduleError, tvm.TVMError):
                mutated_trace = None
            if mutated_trace is None:
                break
            maybe_new = self._replay_schedule(mutated_trace, rand_state)
            if maybe_new is None:
                break
            new_sch = maybe_new

        # 3) cost-model predict (score)
        if not self._cost_model or not self._database:
            return random.random()  # fallback random
        arg_info = ArgInfo.from_entry_func(new_sch.mod, remove_preproc=True)
        candidate = MeasureCandidate(new_sch, arg_info)
        preds = self._cost_model.predict(self._workload_key, [candidate])
        if preds:
            return max(0.0, preds[0])
        return 0.0

    # -------------------------------------------------------------------------
    # Leaves & population
    # -------------------------------------------------------------------------
    def gather_unmeasured_leaves(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Traverse the tree to find leaf nodes (no children) that haven't been measured yet.
        """
        stack = [node]
        leaves = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None and not nd.children:
                wl = None
                if self._database:
                    wl = self._database.commit_workload(nd.schedule.mod)
                if wl not in self._measured_workloads:
                    leaves.append(nd)
            else:
                stack.extend(nd.children)
        return leaves

    def pick_unmeasured_best_leaves(self, root: MCTSNode, batch_size: int) -> List[Schedule]:
        """
        Return up to batch_size unmeasured leaf schedules with highest Q-value
        (Q = total_value / visits).
        """
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
        """
        Return all nodes in the MCTS tree that contain a schedule.
        """
        stack = [root]
        out_nodes = []
        while stack:
            nd = stack.pop()
            if nd.schedule is not None:
                out_nodes.append(nd)
            stack.extend(nd.children)
        return out_nodes

    # -------------------------------------------------------------------------
    # Replay schedules, postprocs, random mutators
    # -------------------------------------------------------------------------
    def _replay_schedule(self, trace: Optional[Trace], rand_state: int) -> Optional[Schedule]:
        """
        Rebuild a Schedule from a trace, ignoring built-in postproc so we can do our own.
        Then apply our postprocs to ensure correctness and constraints.
        """
        if not self._ctx or not self._ctx.mod:
            return None
        mod = self._ctx.mod

        if trace is None:
            # create a fresh schedule
            try:
                sch = Schedule(mod, seed=rand_state or 1, debug_mask="all")
            except (InvalidScheduleError, tvm.TVMError):
                return None
            sch.enter_postproc()
            if not self._apply_postprocs(sch):
                return None
            return sch

        # normal replay
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
        """
        Apply postprocessors to the schedule. If an FFI helper is available,
        use it; otherwise, apply postprocessors in Python.
        """
        if not self._postprocs:
            return True

        ffi_postproc = getattr(_ffi_api, "SearchStrategyApplyPostprocs", None)
        if ffi_postproc is not None:
            try:
                return bool(ffi_postproc(sch, self._postprocs))
            except Exception:
                pass  # Fallback to Python-based postprocessing

        # Python-based postprocessing
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
        # fallback
        return list(self._mutator_probs.keys())[0]

    def _try_mcts_mutation(self, parent_sch: Schedule, rand_state: int) -> Optional[Schedule]:
        """
        Attempt multiple times to produce a mutated schedule that we haven't seen yet.
        """
        attempts = 0
        while attempts <= self.genetic_max_fail_count:
            attempts += 1
            self._mutator_failure_count["total"] += 1
            mut = self._pick_random_mutator(rand_state)
            if mut is None:
                # fallback: just replay parent's trace if no mutators
                child_sch = self._replay_schedule(parent_sch.trace, rand_state)
                if child_sch is not None and self._database:
                    wl = self._database.commit_workload(child_sch.mod)
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
                    wl = self._database.commit_workload(child_sch.mod)
                    if wl not in self._seen_workloads:
                        self._seen_workloads.add(wl)
                        return child_sch
        return None

    # -------------------------------------------------------------------------
    # Cost model scoring
    # -------------------------------------------------------------------------
    def _predict_normalized_score(self, schedules: List[Schedule]) -> List[float]:
        if not schedules or not self._cost_model or not self._database:
            return [0.0] * len(schedules)
        cands = []
        for sch in schedules:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            cands.append(MeasureCandidate(sch, arg_info))
        scores = self._cost_model.predict(self._workload_key, cands)
        return [max(0.0, sc) for sc in scores]

    # -------------------------------------------------------------------------
    # Access to sets of measured and seen workloads
    # -------------------------------------------------------------------------
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
        The set of workload keys we've encountered in generated schedules.
        """
        if self._search_state is not None:
            return self._search_state.seen_workloads
        return set()


# -----------------------------------------------------------------------------
# MCTSTuningState: Orchestrates the MCTS steps, storing dynamic search info
# -----------------------------------------------------------------------------
class MCTSTuningState:
    """
    MCTSTuningState tracks the MCTS root, population, # of trials used,
    best score, etc. The MCTSTuner performs expansions and rollouts; 
    MCTSTuningState decides how to handle each iteration (e.g. picking 
    unmeasured leaves, ranking population, etc.).
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

        # record the mod => workload
        self.mod = context.mod
        self.workload_key = None
        if self.database and self.mod is not None:
            self.workload_key = self.database.commit_workload(self.mod)
            # also store in tuner for cost model usage
            self.tuner._workload_key = self.workload_key

        # main dynamic state
        self.trial_count = 0
        self.num_empty_iters = 0
        self.used_init_population = False
        self.population: List[Tuple[Schedule, bool]] = []
        self.mcts_root: Optional[MCTSNode] = None

        # sets of workloads
        self.measured_workloads: Set[Workload] = set()
        self.seen_workloads: Set[Workload] = set()

        # best score tracking
        self.best_score_so_far = -float("inf")
        self.stale_iter_count = 0
        self.stale_diversity_count = 0
        self.diversity_history: List[float] = []
        self.score_history: List[float] = []
        self.dynamic_pop_size = self.tuner.population_size

        # random seed from context
        rs = context.rand_state
        self.rand_state = rs if rs is not None else 1
        if self.rand_state == 0:
            self.rand_state = 1

    def reset(self) -> None:
        """
        Called from MCTSSearch.post_tuning().
        """

    # -------------------------------------------------------------------------
    # The main method called by the strategy: generate_measure_candidates
    # -------------------------------------------------------------------------
    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """
        Orchestrates MCTS expansions and picks unmeasured best leaves as measure candidates.
        """
        if self.trial_count >= self.max_trials:
            return None

        remaining = self.max_trials - self.trial_count
        batch_size = min(remaining, self.num_trials_per_iter)
        if batch_size <= 0:
            return None

        # On first call, initialize root node & population
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
                logger.info("MCTS: Initialized root with %d child schedules.", len(init_pop))

        # Expand and get updated population
        self.population = self.tuner.explore(
            mcts_root=self.mcts_root,
            population=self.population,
            dynamic_pop_size=self.dynamic_pop_size,
            rand_state=self.rand_state,
        )

        # pick the best unmeasured leaves
        picks = self.tuner.pick_unmeasured_best_leaves(self.mcts_root, batch_size)
        if not picks:
            # no unmeasured => increment empty iter
            self.num_empty_iters += 1
            if self.tuner.verbose >= 1:
                logger.info("MCTS: No unmeasured => empty iters=%d", self.num_empty_iters)
            if self.num_empty_iters >= self.tuner.num_empty_iters_before_early_stop:
                # early stop
                if self.tuner.verbose >= 1:
                    logger.info("MCTS: Stopping early => repeated empty iters.")
                return None
            return None

        # Convert schedules => measure candidates
        measure_cands: List[MeasureCandidate] = []
        for sch in picks:
            arg_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
            measure_cands.append(MeasureCandidate(sch, arg_info))

        if self.tuner.verbose >= 1:
            logger.info(
                "MCTS: generate_measure_candidates => %d cands, total measured so far=%d",
                len(measure_cands), self.trial_count
            )
        return measure_cands

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """
        Called after the measure_candidates have been built and run on the target.
        IMPORTANT: We do NOT call cost_model.update(...) or database.commit_tuning_record(...)
        here. We rely on the default measure callbacks to handle that.
        
        This method only handles MCTS-specific bookkeeping: 
          - marking schedules as measured
          - tracking best performance
          - deciding if we are "stale" or should early-stop
        """
        if self.database is None:
            logger.debug("database is not defined, skipping MCTS measure update.")
            return

        num_measured_now = 0
        best_run_sec = float("inf")

        # Go through each measured candidate
        for cand, res in zip(measure_candidates, results):
            sch = cand.sch
            mod = sch.mod
            wl = self.database.commit_workload(mod)

            # If we got valid timing => track best performance
            if res.run_secs and all(t >= 0 for t in res.run_secs):
                run_sec = sum(res.run_secs) / len(res.run_secs)
                if run_sec < best_run_sec:
                    best_run_sec = run_sec

                # Mark schedule measured in MCTS sets
                self.measured_workloads.add(wl)
                self._mark_schedule_measured(sch)
                num_measured_now += 1

        self.trial_count += num_measured_now

        # Evaluate improvement
        if best_run_sec < float("inf"):
            new_score = 1.0 / best_run_sec
            self.score_history.append(new_score)
            if new_score > self.best_score_so_far + 1e-12:
                self.best_score_so_far = new_score
                self.stale_iter_count = 0
            else:
                self.stale_iter_count += 1
                if self.stale_iter_count >= self.tuner.max_stale_iters and self.tuner.verbose >= 1:
                    logger.info(
                        "MCTS: No improvement => stopping early (stale_iter=%d).",
                        self.stale_iter_count
                    )
        else:
            # no valid runs => score=0
            self.score_history.append(0.0)

        # Check population diversity
        if self.population:
            pop_scores = self._predict_population_scores(self.population)
            diversity = self._check_population_diversity(pop_scores)
            if diversity < self.tuner.diversity_epsilon:
                self.stale_diversity_count += 1
                if self.tuner.verbose >= 1:
                    logger.info(
                        "MCTS: Pop diversity=%.6f < threshold => stale_diversity_count=%d",
                        diversity, self.stale_diversity_count
                    )
                if self.stale_diversity_count >= self.tuner.max_stale_diversity_iters:
                    if self.tuner.verbose >= 1:
                        logger.info("MCTS: Population too homogeneous => stopping early.")
            else:
                self.stale_diversity_count = 0

        # Adaptive population resizing example
        if self.population:
            if self.stale_diversity_count > 0:
                old_size = self.dynamic_pop_size
                self.dynamic_pop_size = max(10, int(self.dynamic_pop_size * 0.9))
                if self.tuner.verbose >= 1:
                    logger.info(
                        "MCTS: Adaptive pop resize: %d -> %d",
                        old_size, self.dynamic_pop_size
                    )
            else:
                self.dynamic_pop_size = min(
                    self.tuner.population_size,
                    self.dynamic_pop_size + 5
                )

        if self.tuner.verbose >= 1:
            logger.info(
                "MCTS: notify_runner_results => measured=%d, total=%d, stale_iter=%d, div_stale=%d",
                num_measured_now, self.trial_count, self.stale_iter_count, self.stale_diversity_count
            )

    # -------------------------------------------------------------------------
    # Initialization of population
    # -------------------------------------------------------------------------
    def _init_population(self) -> List[Tuple[Schedule, bool]]:
        """
        Combine schedules from DB (measured) and random design-space samples (unmeasured).
        """
        # measured from DB
        num_measured_wanted = int(self.tuner.population_size * self.tuner.init_measured_ratio)
        measured_from_db = self._pick_best_from_database(num_measured_wanted)

        # random from design space
        need_rand = max(
            self.tuner.population_size - len(measured_from_db),
            self.tuner.init_min_unmeasured
        )
        unmeasured_rand = self._sample_init_population(need_rand)

        combined = [(sch, True) for sch in measured_from_db] + \
                   [(sch, False) for sch in unmeasured_rand]

        # if not enough unmeasured
        if len(combined) < self.tuner.init_min_unmeasured and self.tuner.verbose >= 1:
            logger.warning("MCTS: Could not collect enough unmeasured schedules.")

        # shuffle => truncate to population size
        random.shuffle(combined)
        if len(combined) > self.tuner.population_size:
            combined = combined[: self.tuner.population_size]

        # update sets
        for (sch, measured_flag) in combined:
            wl = self.database.commit_workload(sch.mod)
            self.seen_workloads.add(wl)
            if measured_flag:
                self.measured_workloads.add(wl)
        return combined

    def _pick_best_from_database(self, num: int) -> List[Schedule]:
        """
        Pick top 'num' schedules from DB. 
        If the database is empty or num<=0 => return [].
        """
        if num <= 0 or not self.database:
            return []
        out = []
        top_records = self.database.get_top_k(self.workload_key, num)
        for rec in top_records:
            sch = self._replay_schedule(rec.trace)
            if sch is not None:
                wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
        return out

    def _replay_schedule(self, trace: Optional[Trace]) -> Optional[Schedule]:
        """
        Basic replay for a trace from DB (like the tuner’s _replay_schedule).
        """
        if not trace or not self.context or not self.context.mod:
            return None
        mod = self.context.mod
        try:
            sch = Schedule(mod, debug_mask="all")
            trace.apply_to_schedule(sch, remove_postproc=True)
        except (InvalidScheduleError, tvm.TVMError):
            return None

        sch.enter_postproc()
        # Try FFI postprocs if available
        if hasattr(_ffi_api, "SearchStrategyApplyPostprocs"):
            try:
                ok = _ffi_api.SearchStrategyApplyPostprocs(sch, self.tuner._postprocs)  # type: ignore
                if not ok:
                    return None
                return sch
            except Exception:
                pass

        # fallback: python-based postprocs
        for proc in self.tuner._postprocs:
            try:
                if not proc.apply(sch):
                    return None
            except (InvalidScheduleError, tvm.TVMError):
                return None
        return sch

    def _sample_init_population(self, num: int) -> List[Schedule]:
        """
        Simple random sampling from the design_spaces.
        Alternatively, can use a parallel C++ approach if desired.
        """
        out = []
        fails = 0
        n_spaces = len(self.design_spaces)
        while len(out) < num and fails < self.tuner.max_fail_count:
            idx = random.randint(0, n_spaces - 1)
            base_sch = self.design_spaces[idx]
            sch = self._replay_schedule(base_sch.trace)
            if sch is not None:
                wl = self.database.commit_workload(sch.mod)
                if wl not in self.seen_workloads:
                    out.append(sch)
                    self.seen_workloads.add(wl)
                else:
                    fails += 1
            else:
                fails += 1
        return out

    # -------------------------------------------------------------------------
    # Mark schedule measured
    # -------------------------------------------------------------------------
    def _mark_schedule_measured(self, sch: Schedule):
        """
        Mark the (sch, measured=True) in the population if it exists there.
        """
        wl = self.database.commit_workload(sch.mod)
        self.measured_workloads.add(wl)
        for i, (pop_sch, was_measured) in enumerate(self.population):
            if pop_sch == sch and not was_measured:
                self.population[i] = (pop_sch, True)

    # -------------------------------------------------------------------------
    # Diversity
    # -------------------------------------------------------------------------
    def _predict_population_scores(self, pop: List[Tuple[Schedule, bool]]) -> List[float]:
        """
        Return predicted cost-model scores for the schedules in pop.
        """
        schs = [p[0] for p in pop]
        if not schs:
            return []
        return self.tuner._predict_normalized_score(schs)

    def _check_population_diversity(self, scores: List[float]) -> float:
        """
        Compute stddev of predicted scores as a measure of diversity.
        """
        if not scores:
            return 0.0
        mean_val = sum(scores) / len(scores)
        var = sum((s - mean_val) ** 2 for s in scores) / len(scores)
        cur_div = math.sqrt(var)
        self.diversity_history.append(cur_div)
        # Keep last 10 diversity values
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        # Adapt the threshold slightly based on recent diversity
        avg_div = sum(self.diversity_history) / len(self.diversity_history)
        self.tuner.diversity_epsilon = 0.5 * avg_div
        return cur_div


# -----------------------------------------------------------------------------
# The final MCTS search strategy to register with MetaSchedule
# -----------------------------------------------------------------------------
@derived_object
class MCTSSearchPyFull(PySearchStrategy):
    """
    An MCTS-based search strategy aligned with MetaSchedule 0.20+ best practices.
    It uses Monte Carlo Tree Search over the design space, letting default measure
    callbacks handle cost model updates and database commits.
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
        num_empty_iters_before_early_stop: int = 5,
        max_stale_iters: int = 6,
        diversity_epsilon: float = 1e-6,
        max_stale_diversity_iters: int = 3,
        trace_commit: bool = True,
        verbose: int = 1,
        # MCTS-specific:
        mcts_ucb_constant: float = 1.41,
        mcts_max_depth: Optional[int] = 20,
        mcts_num_threads: int = 1,
        mcts_num_rollouts_per_expansion: int = 1,
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

        self._ctx: Optional["TuneContext"] = None
        self._postprocs: List[Postproc] = []
        self._mutator_probs: Dict[Mutator, float] = {}
        self.state: Optional[MCTSTuningState] = None

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """
        Called once the TuneContext is ready. We can read space_generator, target, etc.
        """
        self._ctx = context
        if context.space_generator is None:
            raise ValueError("TuneContext.space_generator must be defined.")
        if context.target is None:
            raise ValueError("TuneContext.target must be defined.")

        # collect postprocs
        sg = context.space_generator
        self._postprocs = list(sg.postprocs) if sg.postprocs else []

        # collect mutator_probs
        user_probs = sg.mutator_probs or {}
        for mut, prob_f in user_probs.items():
            p = float(prob_f.value)
            if p > 1e-12:
                self._mutator_probs[mut] = self._mutator_probs.get(mut, 0.0) + p

        # fallback if empty
        target_kind = str(context.target.kind.name)
        if not self._mutator_probs:
            try:
                default_muts = Mutator.create(target_kind)
            except:  # fallback
                default_muts = Mutator.create("llvm")
            if isinstance(default_muts, dict):
                for m, p2 in default_muts.items():
                    self._mutator_probs[m] = float(p2)
            elif isinstance(default_muts, list) and len(default_muts) > 0:
                p2 = 1.0 / len(default_muts)
                for m in default_muts:
                    self._mutator_probs[m] = p2

        # normalize
        total_p = sum(self._mutator_probs.values())
        if total_p > 1e-12:
            for k in self._mutator_probs:
                self._mutator_probs[k] /= total_p

        if self.verbose >= 1:
            logger.info(
                "MCTSSearch: Using target=%s, found #mutators=%d, rand_state=%s",
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
        """
        Called before the tuning process. We create the MCTSTuner and the MCTSTuningState.
        """
        if self.state is not None:
            raise ValueError("MCTSSearch.pre_tuning called without post_tuning after previous run")

        # build MCTSTuner
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
        )

        # build MCTSTuningState
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
            logger.info(
                "MCTSSearch.pre_tuning => max_trials=%d, num_per_iter=%d, #design_spaces=%d",
                max_trials, num_trials_per_iter, len(design_spaces)
            )

    def post_tuning(self) -> None:
        """
        Called after all tuning is finished. We clear state references, 
        optionally log the best result, etc.
        """
        if self.state:
            self.state.reset()
            self.state = None
        if self.verbose >= 1:
            logger.info("MCTSSearch: Tuning finished in post_tuning().")

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """
        Called by the MetaSchedule engine each round to get new schedules for measurement.
        """
        if not self.state:
            logger.warning("MCTSSearch.generate_measure_candidates called before pre_tuning.")
            return None
        return self.state.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """
        Receives measurement results from the runner and updates the MCTS state.
        No cost_model/database commits here (rely on default measure callbacks).
        """
        if self.state:
            self.state.notify_runner_results(measure_candidates, results)

    def clone(self) -> "MCTSSearchPyFull":
        """
        Clone the search strategy. The new copy has no TuningState; it must be
        re-initialized with pre_tuning.
        """
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
        )


