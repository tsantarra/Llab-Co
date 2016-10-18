import cProfile
import io
import logging
import pstats


def profile(func):
    """
    Decorator for profiling individual functions. Creates a .profiler file that can be viewed with
    the pstats module.

    python -m pstats function_name.profiler

    Commands:
        - strip
        - sort time (or other)
        - stats 10 (displays top 10)

    http://stefaanlippens.net/python_profiling_with_pstats_interactive_mode
    """
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profiler"  # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper


if __name__ == "__main__":
    # Initialize profiler
    profiler = cProfile.Profile()
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    # import test
    from domains.multi_agent.coordinated_actions.coordinated_actions import run_coordinated_actions

    # Run code
    profiler.enable()

    run_coordinated_actions()

    profiler.disable()

    # Acquire stats (put into statsString via redirection)
    stats_string = io.StringIO()
    sort_by = 'time'  # calls, cumulative, file, line, module, name, nfl (for name/file/line), pcalls, stdname, time

    pstats.Stats(profiler, stream=stats_string).strip_dirs().sort_stats(sort_by).print_stats()
    results = stats_string.getvalue()
    results = results.split('\n')

    # Print top time intensive calls
    for i in range(4):  # Header lines
        print(results[i])

    for i in range(4, 20):  # Table
        print('\t\t\t'.join(results[i].split()))

