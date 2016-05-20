import cProfile
import io
import logging
import pstats

from domains.cops_and_robbers.cops_and_robbers import carpy_dpthts
from domains.cops_and_robbers.cops_and_robbers_scenario import cops_and_robbers_scenario


def profile(func):
    """
    Decorator for profiling individual functions. Creates a .profile file that can be viewed with
    the pstats module.

    python -m pstats function_name.profile

    Commands:
        - strip
        - sort time (or other)
        - stats 10 (displays top 10)

    http://stefaanlippens.net/python_profiling_with_pstats_interactive_mode
    """
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile"  # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper


if __name__ == "__main__":
    # Initialize profiler
    profiler = cProfile.Profile()
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    # Run code
    profiler.enable()

    carpy_dpthts(cops_and_robbers_scenario)

    profiler.disable()

    # Acquire stats (put into statsString via redirection)
    stats_string = io.StringIO()
    sort_by = 'time'  # calls, cumulative, file, line, module, name, nfl (for name/file/line), pcalls, stdname, time

    pstats.Stats(profiler, stream=stats_string).strip_dirs().sort_stats(sort_by).print_stats()
    results = stats_string.getvalue()
    results = results.split('\n')

    # Print top time intensive calls
    for i in range(10 + 5):  # 5 header lines
        print(results[i])
