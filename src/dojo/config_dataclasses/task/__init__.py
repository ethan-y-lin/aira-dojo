# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class _LazyTask:
    def __init__(self, module_name: str, class_name: str) -> None:
        self.module_name = module_name
        self.class_name = class_name

    def __call__(self, *args, **kwargs):
        module = __import__(self.module_name, fromlist=[self.class_name])
        task_cls = getattr(module, self.class_name)
        return task_cls(*args, **kwargs)

TASK_MAP = {
    "AgentSSLTaskConfig": _LazyTask("dojo.tasks.agentssl.task", "AgentSSLTask"),
    "ActiveProgramSelectionTaskConfig": _LazyTask(
        "dojo.tasks.active_program_selection.task",
        "ActiveProgramSelectionTask",
    ),
}
