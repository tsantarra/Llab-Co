from mdp.distribution import Distribution

from collections import defaultdict


class SparseChineseRestaurantProcessModel:

    def __init__(self, identity, scenario, alpha=1):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario
        self.identity = identity
        self.alpha = alpha
        self.total_observations = 0

        # Lookup info
        self.policies = []
        self.policy_to_index = {}
        self.state_list = []
        self.state_to_index = {}
        self.action_list = []
        self.action_to_index = {}

        # Uniform policy vars
        self.possible_policy_actions = {}

        # The data
        self.observation_counts = []            # policy index -> count
        self.policy_matrix = defaultdict(dict)  # state_id -> dict (policy_id -> action_id)

    def __get_state_index(self, state):
        if state in self.state_to_index:
            return self.state_to_index[state]

        self.state_list.append(state)
        index = len(self.state_list) - 1
        self.state_to_index[state] = index
        return index

    def __get_action_index(self, action):
        if action in self.action_to_index:
            return self.action_to_index[action]

        self.action_list.append(action)
        index = len(self.action_list) - 1
        self.action_to_index[action] = index
        return index

    def add_teammate_model(self, policy):
        """
        Process:
            1. Convert policy to index form.
            2. Check if in policy_to_index:
                a. Yes -> increment observation_counts
                b. No -> add to policy_to_index + policy_matrix(index -> list)
        """
        if type(policy) is list:
            state_action_pairs = policy
        elif type(policy) is dict:
            state_action_pairs = list(policy.items())
        else:
            raise TypeError('Incorrect policy format: ' + str(policy))

        converted_policy = tuple(sorted([(self.__get_state_index(state), self.__get_action_index(action))
                                         for state, action in state_action_pairs]))

        self.total_observations += 1

        # Check if we've encountered this policy before.
        if converted_policy in self.policy_to_index:
            self.observation_counts[self.policy_to_index[converted_policy]] += 1
            return

        # Record observation.
        self.policies.append(converted_policy)
        policy_index = len(self.policies) - 1
        self.policy_to_index[converted_policy] = policy_index
        self.observation_counts.append(1)

        # Add to policy lookup data structure (for predictions).
        for state_index, action_index in converted_policy:
            self.policy_matrix[state_index][policy_index] = action_index

    def prior(self):
        policy_data = [(policy_index, obvs / (self.total_observations + self.alpha))
                       for policy_index, obvs in enumerate(self.observation_counts)]
        policy_data.append((-1, self.alpha / (self.total_observations + self.alpha)))
        return policy_data

    def posterior(self, prior, state, observed_action):
        """
        Returns the posterior distribution after observing an action in a given state.
            P(policy | action) = P(action | policy) * P(policy) / P(action)
        """
        state_index = self.__get_state_index(state)
        action_index = self.__get_action_index(observed_action)

        if state_index not in self.possible_policy_actions:
            self.possible_policy_actions[state_index] = [self.__get_action_index(action)
                                                         for action in self.scenario.actions(state).individual_actions(self.identity)]
        uniform_prob = 1.0 / len(self.possible_policy_actions[state_index])

        # Calculate posterior
        resulting_probs = [probability * uniform_prob
                           if policy_index == -1
                           else (probability
                                 if action_index == self.policy_matrix[state_index][policy_index]
                                 else 0.0)
                           for policy_index, probability in prior]

        # Normalize
        total = sum(resulting_probs)
        resulting_probs = [prob / total for prob in resulting_probs]

        assert abs(sum(resulting_probs) - 1.0) < 10e-6, 'Posterior not normalized: ' + str(sum(resulting_probs))

        return list(pair for pair in zip((pair[0] for pair in prior), resulting_probs) if pair[1] > 0)

    def batch_posterior(self, prior, state_action_pairs):
        """
        Returns the posterior distribution after observing an action in a given state.
            P(policy | action in state) = Π[P(action in state | policy)] * P(policy) / P(all observations)
        """
        policy_indices, probabilities = zip(*prior)

        # Calculate posterior
        for state, observed_action in state_action_pairs:
            state_index = self.__get_state_index(state)
            action_index = self.__get_action_index(observed_action)

            if state_index not in self.possible_policy_actions:
                self.possible_policy_actions[state_index] = [self.__get_action_index(action)
                                                         for action in self.scenario.actions(state).individual_actions(self.identity)]
            uniform_prob = 1.0 / len(self.possible_policy_actions[state_index])

            probabilities = [probability * uniform_prob
                             if policy_index == -1
                             else (probability
                                   if probability > 0 and action_index == self.policy_matrix[state_index][policy_index]
                                   else 0)
                             for policy_index, probability in zip(policy_indices, probabilities)]

        # Normalize
        total = sum(probabilities)
        return list((index, probability / total)
                    for index, probability in zip(policy_indices, probabilities) if probability > 0)

    def get_action_distribution(self, state, policy_distribution):
        state_index = self.__get_state_index(state)
        if state_index not in self.possible_policy_actions:
            self.possible_policy_actions[state_index] = [self.__get_action_index(action)
                                                         for action in self.scenario.actions(state).individual_actions(self.identity)]

        action_index_distribution = {action_index: 0.0 for action_index in self.possible_policy_actions[state_index]}
        uniform_prob = 1.0 / len(self.possible_policy_actions[state_index])

        assert abs(sum(val for pol, val in policy_distribution) - 1.0) < 10e-6, \
            'Policy distribution not normalized: ' + str(sum(val for pol, val in policy_distribution))

        for policy_index, policy_probability in policy_distribution:
            if policy_index == -1:
                for action_index in action_index_distribution:
                    action_index_distribution[action_index] += policy_probability * uniform_prob
            else:
                if state_index not in self.policy_matrix:
                    print('not correct state index: ' + str(state_index))
                if policy_index not in self.policy_matrix[state_index]:
                    print('not correct policy index: ' + str(policy_index) + '\n' + str(self.policy_matrix[state_index]))
                action_index = self.policy_matrix[state_index][policy_index]
                if action_index not in action_index_distribution:
                    print('not correct action index: ' + str(action_index) + '\n' + str(action_index_distribution))
                action_index_distribution[self.policy_matrix[state_index][policy_index]] += policy_probability

        assert abs(sum(action_index_distribution.values()) - 1.0) < 10e-6, 'Action predictions not normalized.'

        return Distribution({self.action_list[action_index]: probability
                             for action_index, probability in action_index_distribution.items()})


class ChineseRestaurantProcessModel:

    def __init__(self, identity, scenario, policy_state_order, policy_actions, alpha=1):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario
        self.identity = identity
        self.alpha = alpha
        self.total_obvs = 0

        # The data
        self.observation_counts = list()  # policy index -> count
        self.policy_matrix = list()  # policy index -> policy list (state index -> action index)
        self.action_list = list(set(action for action_set in policy_actions for action in action_set))

        # Lookup info
        self.policy_to_index = dict()
        self.state_to_index = {state: index for index, state in enumerate(policy_state_order)}
        self.action_to_index = {action: index for index, action in enumerate(self.action_list)}

        # Uniform policy vars
        self.possible_policy_actions = [[self.action_to_index[action] for action in action_set]
                                        for action_set in policy_actions]

    def add_teammate_model(self, policy):
        """
        Process:
            1. Convert policy to index form.
            2. Check if in policy_to_index:
                a. Yes -> increment observation_counts
                b. No -> add to policy_to_index + policy_matrix(index -> list)
        """
        assert len(policy) == len(self.state_to_index), 'Incorrect policy length given: {0} vs {1} \n{2}'.format(
            len(policy), len(self.state_to_index), policy)

        # Convert policy and check if already encountered
        action_index_policy = tuple([self.action_to_index[action] for state, action in policy])
        if action_index_policy in self.policy_to_index:
            self.observation_counts[self.policy_to_index[action_index_policy]] += 1
            self.total_obvs += 1
            return

        # Add to dataset
        policy_index = len(self.policy_to_index)
        self.policy_to_index[action_index_policy] = policy_index
        self.observation_counts.append(1)
        self.total_obvs += 1
        self.policy_matrix.append(action_index_policy)

    def prior(self):
        policy_data = [(policy_index, obvs / (self.total_obvs + self.alpha))
                       for policy_index, obvs in enumerate(self.observation_counts)]
        policy_data.append((-1, self.alpha / (self.total_obvs + self.alpha)))
        return policy_data

    def posterior(self, prior, state, observed_action):
        """
        Returns the posterior distribution after observing an action in a given state.
            P(policy | action) = P(action | policy) * P(policy) / P(action)
        """
        state_index = self.state_to_index[state]
        action_index = self.action_to_index[observed_action]
        uniform_prob = 1.0 / len(self.possible_policy_actions[state_index])

        # Calculate posterior
        resulting_probs = [probability * uniform_prob
                           if policy_index == -1
                           else (probability
                                 if action_index == self.policy_matrix[policy_index][state_index]
                                 else 0.0)
                           for policy_index, probability in prior]

        # Normalize
        total = sum(resulting_probs)
        resulting_probs = [prob / total for prob in resulting_probs]

        assert abs(sum(resulting_probs) - 1.0) < 10e-6, 'Posterior not normalized: ' + str(sum(resulting_probs))

        return list(pair for pair in zip((pair[0] for pair in prior), resulting_probs) if pair[1] > 0)

    def batch_posterior(self, prior, state_action_pairs):
        """
        Returns the posterior distribution after observing an action in a given state.
            P(policy | action in state) = Π[P(action in state | policy)] * P(policy) / P(all observations)
        """
        indices, probabilities = zip(*prior)

        # Calculate posterior
        for state, observed_action in state_action_pairs:
            state_index = self.state_to_index[state]
            action_index = self.action_to_index[observed_action]
            uniform_prob = 1.0 / len(self.possible_policy_actions[state_index])

            probabilities = [probability * uniform_prob
                             if policy_index == -1
                             else (probability
                                   if probability > 0 and action_index == self.policy_matrix[policy_index][state_index]
                                   else 0)
                             for policy_index, probability in zip(indices, probabilities)]

        # Normalize
        total = sum(probabilities)
        return list(
            (index, probability / total) for index, probability in zip(indices, probabilities) if probability > 0)

    def get_action_distribution(self, state, policy_distribution):
        state_index = self.state_to_index[state]
        action_index_distribution = {action_index: 0.0 for action_index in self.possible_policy_actions[state_index]}
        uniform_prob = 1.0 / len(action_index_distribution)

        assert abs(sum(val for pol, val in policy_distribution) - 1.0) < 10e-6, \
            'Policy distrubition not normalized: ' + str(sum(val for pol, val in policy_distribution))

        for policy_index, policy_probability in policy_distribution:
            if policy_index == -1:
                for action_index in action_index_distribution:
                    action_index_distribution[action_index] += policy_probability * uniform_prob
            else:
                action_index_distribution[self.policy_matrix[policy_index][state_index]] += policy_probability

        return Distribution({self.action_list[action_index]: probability
                             for action_index, probability in action_index_distribution.items()})
