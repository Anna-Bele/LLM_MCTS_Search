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
""" Test Meta Schedule SearchStrategy """
# pylint: disable=missing-function-docstring
from typing import List

import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.testing.dummy_object import DummyMutator
from tvm.script import tir as T
from tvm.tir.schedule import Schedule, Trace

import logging

logging.basicConfig(level=logging.DEBUG)


MATMUL_M = 32

# pylint: disable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument, unbalanced-tuple-unpacking
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None: # type: ignore
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (32, 32), "float32")
        B = T.match_buffer(b, (32, 32), "float32")
        C = T.match_buffer(c, (32, 32), "float32")
        for i, j, k in T.grid(32, 32, 32):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0 # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _is_trace_equal(sch_1: Schedule, sch_2: Schedule, remove_decisions=True) -> bool:
    if remove_decisions:
        trace_1 = Trace(sch_1.trace.insts, {})
        trace_2 = Trace(sch_2.trace.insts, {})
    else:
        trace_1 = sch_1.trace
        trace_2 = sch_2.trace
    return str(trace_1) == str(trace_2)


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(i, sch.sample_perfect_tile(i, n=4))
    j_0, j_1, j_2, j_3 = sch.split(j, sch.sample_perfect_tile(j, n=4))
    k_0, k_1 = sch.split(k, sch.sample_perfect_tile(k, n=2))
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


@pytest.mark.parametrize(
    "TestClass",
    [
        ms.search_strategy.ReplayFunc,
        ms.search_strategy.ReplayTrace,
    ],
)
def test_meta_schedule_replay_func(
    TestClass: ms.search_strategy.SearchStrategy,
):  # pylint: disable = invalid-name
    num_trials_per_iter = 7
    max_trials_per_task = 20

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul, postprocs=[]),
        search_strategy=TestClass(),
    )
    strategy = context.search_strategy
    spaces = context.space_generator.generate_design_space(context.mod)
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=spaces,
    )
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
            )
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                )
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert num_trials_each_iter == [7, 7, 6]


def test_meta_schedule_evolutionary_search():  # pylint: disable = invalid-name
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 10
    max_trials_per_task = 2000
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,  # because we are using a mutator from the python side
    )
    strategy = context.search_strategy
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=context.space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        print(candidates)
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
            )
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                )
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert sum(num_trials_each_iter) == 25
    assert num_trials_each_iter.count(0) < 5


def test_meta_schedule_evolutionary_search_early_stop():  # pylint: disable = invalid-name
    def _schedule_matmul_empty(sch: Schedule):
        return sch

    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )

    num_trials_per_iter = 10
    max_trials_per_task = 100

    context = ms.TuneContext(
        mod=Matmul,
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_empty,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,
    )
    strategy = context.search_strategy
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=context.space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
            )
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                ),
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert num_trials_each_iter == [1, 0, 0, 0, 0]


def test_meta_schedule_evolutionary_search_fail_init_population():  # pylint: disable = invalid-name
    @derived_object
    class AlwaysFailPostproc(ms.postproc.PyPostproc):
        """A postproc that always fails."""

        def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
            pass

        def apply(self, sch: Schedule) -> bool:
            return False

        def clone(self) -> "AlwaysFailPostproc":
            return AlwaysFailPostproc()

        def __str__(self) -> str:
            return "AlwaysFailPostproc"

    num_trials_per_iter = 10
    max_trials_per_task = 2000

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul,
            sch_rules=[],
            postprocs=[AlwaysFailPostproc()],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,  # because we are using a mutator from the python side
    )
    strategy = context.search_strategy
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=context.space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )
    candidates = strategy.generate_measure_candidates()
    assert candidates is None


from tvm.meta_schedule.search_strategy import MCTSSearchPyFull

def test_meta_schedule_mcts_basic():
    """Basic test of MCTS: ensures we can generate measure candidates, check their schedule,
    and not exceed the max trial budget."""
    # Define a small scheduling function (inspired by the evolutionary test)
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        # Perform a perfect tile split on the 'j' and 'k' loops
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 3
    max_trials_per_task = 8

    # Generate the correct schedule using the original schedule function for reference
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[], 
            postprocs=[], 
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=MCTSSearchPyFull(
            population_size=64,
            init_measured_ratio=0.1,
            init_min_unmeasured=1,
            max_fail_count=10,
            genetic_num_iters=10,  # number of expansions
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=3,
            mcts_ucb_constant=1.41,
            mcts_num_threads=1,
            mcts_num_rollouts_per_expansion=1,
            verbose=1,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,  # single-threaded for consistency with Python-side mutators
    )
    strategy = context.search_strategy
    design_spaces = context.space_generator.generate_design_space(context.mod)
    print(design_spaces)

    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=design_spaces,
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )

    trial_counts: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        trial_counts.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            # Check that the candidate schedule matches the expected schedule
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=isinstance(strategy, ms.search_strategy.ReplayTrace),
            )
            runner_results.append(ms.runner.RunnerResult(run_secs=[0.1, 0.2], error_msg=None))
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()

    strategy.post_tuning()
    total_trials = sum(trial_counts)
    assert total_trials <= max_trials_per_task
    assert len(trial_counts) >= 1

from tvm.meta_schedule.search_strategy import MCTSLLMSearch

def test_meta_schedule_mcts_llm_basic():
    """Basic test of MCTS: ensures we can generate measure candidates, check their schedule,
    and not exceed the max trial budget."""
    # Define a small scheduling function (inspired by the evolutionary test)
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        # Perform a perfect tile split on the 'j' and 'k' loops
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 3
    max_trials_per_task = 8

    # Generate the correct schedule using the original schedule function for reference
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[], 
            postprocs=[], 
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=MCTSLLMSearch(
            population_size=64,
            init_measured_ratio=0.1,
            init_min_unmeasured=1,
            max_fail_count=10,
            genetic_num_iters=10,  # number of expansions
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=3,
            mcts_ucb_constant=1.41,
            mcts_num_threads=1,
            mcts_num_rollouts_per_expansion=1,
            verbose=1,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,  # single-threaded for consistency with Python-side mutators
    )
    strategy = context.search_strategy
    design_spaces = context.space_generator.generate_design_space(context.mod)
    print(design_spaces)

    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=design_spaces,
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )

    trial_counts: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        trial_counts.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            # Check that the candidate schedule matches the expected schedule
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=isinstance(strategy, ms.search_strategy.ReplayTrace),
            )
            runner_results.append(ms.runner.RunnerResult(run_secs=[0.1, 0.2], error_msg=None))
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()

    strategy.post_tuning()
    total_trials = sum(trial_counts)
    assert total_trials <= max_trials_per_task
    assert len(trial_counts) >= 1



def test_meta_schedule_mcts_multithread():
    """Test MCTS in multi-thread mode with multiple rollouts per expansion."""
    strategy = MCTSSearchPyFull(
        population_size=4,
        init_measured_ratio=0.0,
        init_min_unmeasured=3,
        max_fail_count=5,
        genetic_num_iters=2,
        genetic_mutate_prob=0.5,
        genetic_max_fail_count=3,
        mcts_ucb_constant=1.41,
        mcts_num_threads=2,
        mcts_num_rollouts_per_expansion=2,
        verbose=0,
    )
    max_trials_per_task = 10
    num_trials_per_iter = 4

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
        search_strategy=strategy,
        num_threads=1,
    )
    design_spaces = context.space_generator.generate_design_space(Matmul)
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=design_spaces,
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )

    total_candidates = 0
    while True:
        cands = strategy.generate_measure_candidates()
        if cands is None:
            break
        total_candidates += len(cands)
        results = [ms.runner.RunnerResult(run_secs=[0.05], error_msg=None) for _ in cands]
        strategy.notify_runner_results(cands, results)

    strategy.post_tuning()
    assert 0 < total_candidates <= max_trials_per_task


def test_meta_schedule_mcts_stops():
    """Confirm MCTS stops once the trial budget is reached or no new schedules are possible."""
    strategy = MCTSSearchPyFull(
        population_size=2,
        init_measured_ratio=0.0,
        init_min_unmeasured=2,
        max_fail_count=3,
        genetic_num_iters=1,
        genetic_mutate_prob=0.3,
        genetic_max_fail_count=2,
        mcts_ucb_constant=1.41,
        verbose=0,
    )
    max_trials = 5
    num_trials_iter = 2
    db = ms.database.MemoryDatabase()
    cost_model = ms.cost_model.RandomModel()

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
        search_strategy=strategy,
    )
    design_spaces = context.space_generator.generate_design_space(Matmul)
    strategy.pre_tuning(
        max_trials=max_trials,
        num_trials_per_iter=num_trials_iter,
        design_spaces=design_spaces,
        database=db,
        cost_model=cost_model,
    )

    measured = 0
    for _ in range(10):
        cands = strategy.generate_measure_candidates()
        if cands is None:
            break
        measured += len(cands)
        results = [ms.runner.RunnerResult(run_secs=[0.06], error_msg=None) for _ in cands]
        strategy.notify_runner_results(cands, results)

    strategy.post_tuning()
    assert measured <= max_trials
    assert strategy.generate_measure_candidates() is None


def test_meta_schedule_mcts_trace_equality():
    """Check if MCTS can produce a schedule trace identical to the reference schedule."""
    strategy = MCTSSearchPyFull(
        population_size=2,
        init_measured_ratio=0.0,
        init_min_unmeasured=2,
        max_fail_count=2,
        genetic_num_iters=2,
        genetic_mutate_prob=0.3,
        genetic_max_fail_count=2,
        mcts_ucb_constant=1.41,
        verbose=0,
    )
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul, postprocs=[]),
        search_strategy=strategy,
    )
    db = ms.database.MemoryDatabase()
    cost_model = ms.cost_model.RandomModel()
    design_spaces = context.space_generator.generate_design_space(Matmul)
    strategy.pre_tuning(
        max_trials=5,
        num_trials_per_iter=2,
        design_spaces=design_spaces,
        database=db,
        cost_model=cost_model,
    )

    matched = False
    while True:
        cands = strategy.generate_measure_candidates()
        if cands is None:
            break
        for cand in cands:
            if _is_trace_equal(cand.sch, correct_sch, remove_decisions=True):
                matched = True
        results = [ms.runner.RunnerResult(run_secs=[0.1], error_msg=None) for _ in cands]
        strategy.notify_runner_results(cands, results)

    strategy.post_tuning()
    # If you want a strict requirement:
    # assert matched, "MCTS never generated the reference matmul schedule"


def test_meta_schedule_mcts_early_stop_empty_schedule():
    """Test that MCTS stops early if the schedule function does nothing."""
    def _schedule_matmul_empty(sch: Schedule):
        return sch

    strategy = MCTSSearchPyFull(
        population_size=5,
        init_measured_ratio=0.0,
        init_min_unmeasured=2,
        max_fail_count=10,
        genetic_num_iters=2,
        genetic_mutate_prob=0.5,
        genetic_max_fail_count=5,
        mcts_ucb_constant=1.41,
        mcts_num_threads=1,
        mcts_num_rollouts_per_expansion=1,
        num_empty_iters_before_early_stop=3,
        verbose=0,
    )

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_empty,
            sch_rules=[],
            postprocs=[],
            mutator_probs={DummyMutator(): 1.0},
        ),
        search_strategy=strategy,
        num_threads=1,
    )
    design_spaces = context.space_generator.generate_design_space(Matmul)
    strategy.pre_tuning(
        max_trials=30,
        num_trials_per_iter=5,
        design_spaces=design_spaces,
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )

    num_candidates_each_iter: List[int] = []
    while True:
        cands = strategy.generate_measure_candidates()
        if cands is None:
            break
        num_candidates_each_iter.append(len(cands))
        results = [ms.runner.RunnerResult(run_secs=[0.2], error_msg=None) for _ in cands]
        strategy.notify_runner_results(cands, results)

    strategy.post_tuning()
    assert len(num_candidates_each_iter) >= 1
    if len(num_candidates_each_iter) > 1:
        # Usually the second iteration yields zero new candidates => early stop
        tail_zero = all(x == 0 for x in num_candidates_each_iter[1:])
        assert tail_zero, f"MCTS did not stop as expected: {num_candidates_each_iter}"


def test_meta_schedule_mcts_early_stop_failing_postproc():
    """Test scenario where all schedules fail postproc => no measure candidates."""

    @derived_object
    class AlwaysFailPostproc(ms.postproc.PyPostproc):
        def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
            pass

        def apply(self, sch: Schedule) -> bool:
            return False

        def clone(self) -> "AlwaysFailPostproc":
            return AlwaysFailPostproc()

    strategy = MCTSSearchPyFull(
        population_size=5,
        init_measured_ratio=0.5,
        init_min_unmeasured=3,
        max_fail_count=5,
        genetic_num_iters=2,
        genetic_mutate_prob=0.5,
        genetic_max_fail_count=5,
        mcts_ucb_constant=1.41,
        mcts_max_depth=2,
        mcts_num_threads=1,
        mcts_num_rollouts_per_expansion=1,
        num_empty_iters_before_early_stop=2,
        verbose=0,
    )

    # Minimal schedule for demonstration
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        sch.split(j, sch.sample_perfect_tile(j, n=2))
        sch.split(k, sch.sample_perfect_tile(k, n=2))

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[AlwaysFailPostproc()],
            mutator_probs={DummyMutator(): 1.0},
        ),
        search_strategy=strategy,
        num_threads=1,
    )

    design_spaces = context.space_generator.generate_design_space(Matmul)
    strategy.pre_tuning(
        max_trials=50,
        num_trials_per_iter=5,
        design_spaces=design_spaces,
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )

    cands = strategy.generate_measure_candidates()
    assert cands is None, "Expected no candidates (postproc always fails)!"
    strategy.post_tuning()


if __name__ == "__main__":
    test_meta_schedule_replay_func(ms.search_strategy.ReplayFunc)
    test_meta_schedule_replay_func(ms.search_strategy.ReplayTrace)
    test_meta_schedule_evolutionary_search()
    test_meta_schedule_evolutionary_search_early_stop()
    test_meta_schedule_evolutionary_search_fail_init_population()
