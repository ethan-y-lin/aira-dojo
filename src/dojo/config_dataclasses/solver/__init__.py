# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dojo.solvers.greedy import Greedy
from dojo.solvers.mcts import MCTS
from dojo.solvers.evo import Evolutionary
from dojo.solvers.coda_active_search import CODAActiveSearch
from dojo.config_dataclasses.solver.coda_active_search import CODAActiveSearchSolverConfig

SOLVER_MAP = {
    "GreedySolverConfig": Greedy,
    "MCTSSolverConfig": MCTS,
    "EvolutionarySolverConfig": Evolutionary,
    "CODAActiveSearchSolverConfig": CODAActiveSearch,
}
