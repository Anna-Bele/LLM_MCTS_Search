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
"""Evolutionary Search Strategy"""
from tvm._ffi import register_object

from .. import _ffi_api
from .search_strategy import SearchStrategy


@register_object("meta_schedule.EvolutionarySearch")
class EvolutionarySearch(SearchStrategy):
    """
    Replay Trace Search Strategy is a search strategy that always replays the trace by removing its
    decisions so that the decisions would be randomly re-generated.

    Parameters
    ----------
    population_size : int
        The initial population of traces from measured samples and randomly generated samples.
    init_measured_ratio : int
        The ratio of measured samples in the initial population.
    init_min_unmeasured : int
        The minimal size of unmeasured population in the initial sampling.
    max_fail_count : int
        The maximum number of failure during initial sampling.
    genetic_num_iters : int
        The number of iterations for genetic algorithm.
    genetic_mutate_prob : float
        The probability of mutation.
    genetic_max_fail_count : int
        The maximum number to retry mutation.
    eps_greedy : float
        The ratio of greedy selected samples in the final picks.
    """

    population_size: int
    init_measured_ratio: int
    init_min_unmeasured: int
    genetic_num_iters: int
    genetic_mutate_prob: float
    genetic_max_fail_count: int
    eps_greedy: float

    def __init__(
        self,
        *,
        population_size: int = 3,
        init_measured_ratio: float = 0,
        init_min_unmeasured: int = 3,
        max_fail_count: int = 5,
        genetic_num_iters: int = 47,
        genetic_mutate_prob: float = 0.85,
        genetic_max_fail_count: int = 10,
        eps_greedy: float = 0.05,
    ) -> None:
        """Constructor"""
        self.__init_handle_by_constructor__(
            _ffi_api.SearchStrategyEvolutionarySearch,  # type: ignore # pylint: disable=no-member
            population_size,
            init_measured_ratio,
            init_min_unmeasured,
            max_fail_count,
            genetic_num_iters,
            genetic_mutate_prob,
            genetic_max_fail_count,
            eps_greedy,
        )
