"""Sandboxed Python executor.

From the article's "Base Model Performance" section: a sandboxed Python executor
with a timeout, using exec() with restricted globals. It runs the extracted code
plus the problem's test harness and reports pass/fail.

The article runs tests "in milliseconds"; we run each test in a separate process
with a timeout so a bad generation (infinite loop, crash) can't hang training.
"""

import multiprocessing as mp


def _run(code: str, test: str, entry_point: str, queue):
    """Child-process worker: exec code + test, push ('ok'/'fail', error)."""
    try:
        env = {"__name__": "__main__"}
        exec(code, env)
        exec(test, env)
        # HumanEval tests define `check(candidate)`; call it on the function.
        env["check"](env[entry_point])
        queue.put((True, None))
    except Exception as e:  # noqa: BLE001 - any failure means the test didn't pass
        queue.put((False, repr(e)))


class HumanEvalExecutor:
    """Runs extracted code against a HumanEval test, with a timeout."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def execute_test(self, code: str, test: str, entry_point: str):
        """Return (passed: bool, error: str | None)."""
        queue = mp.Queue()
        proc = mp.Process(target=_run, args=(code, test, entry_point, queue))
        proc.start()
        proc.join(self.timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False, "timeout"

        if not queue.empty():
            return queue.get()
        return False, "no result"
