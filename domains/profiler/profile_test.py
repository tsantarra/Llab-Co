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
    from domains.multi_agent.recipe_sat.recipe_sat import centralized_recipe_sat

    # Run code
    profiler.enable()

    centralized_recipe_sat()

    profiler.disable()

    # Acquire stats (put into statsString via redirection)
    stats_string = io.StringIO()
    sort_by = 'time'  # calls, cumulative, file, line, module, name, nfl (for name/file/line), pcalls, stdname, time

    pstats.Stats(profiler, stream=stats_string).strip_dirs().sort_stats(sort_by).print_stats()
    results = stats_string.getvalue()
    results = results.split('\n')

    # Header lines
    for i in range(4):
        print(results[i])

    cols = results[4].split()
    padding = 3
    all_targets = [(max(len(row.split()[col]) for row in results[4:] if (len(row.split()) >= len(cols))) + padding)
                   for col in range(len(cols))]

    # Table
    for i in range(4, 25):
        items = results[i].split()
        items = items[0:len(cols)-1] + [''.join(items[len(cols)-1:])]  # rejoin spaces in built-in method signatures

        if i > 4:
            ncalls = int(items[0].split('/')[0])
            tottime = float(items[1])
            cumtime = float(items[3])
            items[2] = "{0:.6f}".format(tottime/ncalls)
            items[4] = "{0:.6f}".format(cumtime/ncalls)

        print(''.join(item + ' ' * (target - len(item)) for item, target in zip(items, all_targets)))


