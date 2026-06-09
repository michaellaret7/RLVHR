"""Sandboxed Python executor.

From the article's "Base Model Performance" section: a sandboxed Python executor
with a timeout, using exec() with restricted globals. It runs the extracted code
plus the problem's test harness and reports pass/fail.

The article runs tests "in milliseconds"; we run each test in a separate process
with a timeout so a bad generation (infinite loop, crash) can't hang training.
"""

import multiprocessing as mp
import time

def _run(code: str, test: str, entry_point: str, queue):
    """Child-process worker: exec code + test, push ('ok'/'fail', error)."""
    try:
        env = {"__name__": "__main__"}

        # Add the function objects to the env so the entry_point is available
        # Add the returned code object to the env and the check function object to the env
        # The code func obj is whats returned by the llm
        exec(code, env)
        exec(test, env)

        # HumanEval tests define `check(candidate)`; call it on the function.
        # Thats why entry point is necessary, because we need to call the check function with the entry point function name
        # This line right here is what calls the actual written func obj from the llm through the check function obj
        # Simple example: env("def x(n): return n*n", env) to run it you would call env["x"](2) the result would be 4
        env["check"](env[entry_point])

        queue.put((True, None)) # add the test reuslt to the queue
        
    except Exception as e:  # noqa: BLE001 - any failure means the test didn't pass
        queue.put((False, repr(e))) # add the test result to the queue


class HumanEvalExecutor:
    """Runs extracted code against a HumanEval test, with a timeout."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    # Take the output code from the llm and the test dunc object from the corresponding problem
    # and test to see if the code passes the test
    def execute_test(self, code: str, test: str, entry_point: str):
        """Return (passed: bool, error: str | None)."""
        queue = mp.Queue()

        proc = mp.Process(target=_run, args=(code, test, entry_point, queue))

        # start the process of running the run func
        proc.start()
        
        # wait for the process to finish or timeout
        proc.join(self.timeout)
        
        if proc.is_alive():
            # if the process is still running after the timeout period, terminate it
            proc.terminate()
            proc.join()

            return False, "timeout" # return false and the error message

        if not queue.empty():
            return queue.get()

        return False, "no result"

    def execute_batch(self, items):
        """Run several (code, test, entry_point) tests concurrently.

        Each test still runs in its own process so a runaway generation can be
        killed on timeout; launching them all together overlaps the spawn + run
        cost instead of paying it once per completion. The timeout is shared, so
        total wall-time is ~timeout rather than len(items) * timeout.

        Returns a list of (passed, error) in the same order as `items`.
        """
        procs = []

        for code, test, entry_point in items:
            queue = mp.Queue()
            proc = mp.Process(target=_run, args=(code, test, entry_point, queue))
            proc.start()
            procs.append((proc, queue))

        start = time.monotonic()
        results = []

        for proc, queue in procs:
            remaining = max(0.0, self.timeout - (time.monotonic() - start))
            proc.join(remaining)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                results.append((False, "timeout"))
                
            elif not queue.empty():
                results.append(queue.get())
            else:
                results.append((False, "no result"))

        return results

if __name__ == "__main__":
    from environment.data import load_humaneval

    train_problems, eval_problems, train_prompts = load_humaneval()

    executor = HumanEvalExecutor()

    code = eval_problems[0]["prompt"] + eval_problems[0]["canonical_solution"]

    passed, error = executor.execute_test(
        code,
        eval_problems[0]["test"],
        eval_problems[0]["entry_point"],
    )

    print(passed, error)
