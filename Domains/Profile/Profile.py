import cProfile
import io
import pstats

from Domains.Cops_and_Robbers.CopsAndRobbers import carpy_dpthts
from Domains.Cops_and_Robbers.CopsAndRobbersScenario import cops_and_robbers_scenario
import logging

from Domains.Grid.GridScenario import grid_scenario

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
