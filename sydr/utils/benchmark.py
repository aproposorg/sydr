
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(self, *args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter_ns()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        # print(f'Function {func.__name__} Took {total_time:.4f} ns')

        # Save the results
        self.channelResults.append(self.prepareProfilingResults(f"{func.__name__}", total_time))

        return result
    return timeit_wrapper
