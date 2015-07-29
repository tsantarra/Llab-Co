from GridTransitionManager import GridTransitionManager
from GridActionManager import GridActionManager
from TestGrid import initialize, GridTestBFS, GridTestMCTS, GridTestVI
from solvers.BreadthFirstSearch import BreadthFirstSearch
from graph.State import State

import cProfile
import pstats
import io

if __name__ == "__main__":
    #Initialize profiler
    profiler = cProfile.Profile()

    #Initialize scenario
    scenario = initialize()

    #Run code
    profiler.enable()
    #GridTestBFS(scenario)
    #GridTestMCTS(scenario)
    GridTestVI(scenario)
    profiler.disable()

    #Acquire stats (put into statsString via redirection)
    statsString = io.StringIO()
    sortby = 'time' 
    #calls, cumulative, file, line, module, name, 
    #nfl (for name/file/line), pcalls, stdname, time

    pstats.Stats(profiler, stream=statsString).strip_dirs().sort_stats(sortby).print_stats()
    
    results = statsString.getvalue()
    results = results.split('\n')
    
    #Print top time intensive calls
    for i in range(10+5): #5 header lines
        print(results[i])



