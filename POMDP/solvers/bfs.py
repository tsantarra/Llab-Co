def breadth_first_search(initial_state, scenario):
    """
    Searches for the goal state using BFS.

    Args:
        goal: The goal state to be achieved.
        initial_state: The starting state of the planner.
        scenario: The mechanics of the environment.

    Returns:
        A plan, if one is found; otherwise, None.

    """

    # Copy initial state to work with.
    state = initial_state.copy()

    # Initialize queue of state/plan pairs.
    queue = [(state, [])]

    # States covered in search
    states_covered = set()

    # While states remain to be explored, search the state/plan queue.
    while queue: 
        # Retrieve state/plan to examine.
        current_state, plan = queue.pop(0)

        # If goal reached, return plan. Here, goal may only specify certain features, so we check if
        # current_state is covered by goal's target feature values.
        if scenario.end(current_state):
            return plan

        #  Expand state by one step. Add resulting state/plan pairs to queue.
        for action in scenario.actions(current_state):
            # Get new state after action
            new_state = scenario.transition(current_state, action).sample()

            # Convert state to hashable representation. Check if search has already covered said state.
            state_rep = tuple(new_state.items())
            if state_rep not in states_covered:
                # If not yet reached, add to covered states, adjust plan, and add to queue.
                states_covered.add(state_rep)
                new_plan = list(plan) + [action]
                queue += [(new_state, new_plan)]

    # No successful plan was found.
    return None
