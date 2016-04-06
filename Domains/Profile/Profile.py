import cProfile
import io
import pstats

from MDP.Domains.Grid.Grid import grid_test_dpthts

from Domains.Grid.GridScenario import grid_scenario

if __name__ == "__main__":
    # Initialize profiler
    profiler = cProfile.Profile()

    # Run code
    profiler.enable()

    grid_test_dpthts(grid_scenario)
    # grid_test_thts(grid_scenario)
    # grid_test_mcts(grid_scenario)
    # grid_test_vi(grid_scenario)

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
