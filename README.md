<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# LLM MCTS Strategy
In our project, we use TVM since it is an open source compiler stack for deep learning systems with [Apache-2.0](LICENSE) license. Detailed implementations of this project are included in the folder python/tvm/meta_schedule/search_strategy.

To run this repo, follow these steps:
1. Install TVM and configure the environment as detailed in TVM's documentation https://tvm.apache.org/docs/install/index.html
2. Instead of using the default strategy, create the MCTS+LLM search
strategy object by

mcts_strategy = MCTSSearchPyFull(
    population_size=3,
    init_measured_ratio=0,
    init_min_unmeasured=3,
    max_fail_count=20,
    genetic_num_iters=3,
    genetic_mutate_prob=0.85,
    genetic_max_fail_count=2,
    num_empty_iters_before_early_stop=100,
    max_stale_iters=60,
    diversity_epsilon=1e-6,
    max_stale_diversity_iters=30,
    trace_commit=True,
    mcts_ucb_constant=1.41,
    mcts_max_depth=2000,
    mcts_num_threads=1,
    mcts_num_rollouts_per_expansion=1,
    use_llm=True,
    llm_budget=600,
    llm_model_name="API_MODEL_NAME",
)

If you want to run the pure MCTS search, set use_llm = False so you do not enable LLM.