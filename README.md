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

# LLM-Guided MCTS for Compiler Optimization
In our project, we use TVM since it is an open source compiler stack for deep learning systems with [Apache-2.0](LICENSE) license. 

To run this repo, follow these steps:
1. Clone this repo. Configure the environment as detailed in TVM's documentation https://tvm.apache.org/docs/install/index.html
2. Instead of using the default strategy, create the LLM guided MCTS search strategy object by

```
llm_mcts_strategy = MCTSSearchPyFull(
    use_llm=True,
    llm_budget=600,
    llm_model_name="API_MODEL_NAME",
)
```

If you want to run the pure MCTS search, set use_llm = False so you do not enable LLM.

To use the function tune_tir for tuning, pass llm_mcts_strategy as a parameter of tune_tir, like

```
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
    strategy=llm_mcts_strategy,
)
```