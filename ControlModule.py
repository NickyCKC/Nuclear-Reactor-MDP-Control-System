# Import required dependencies
import numpy as np
import mdptoolbox


class ControlModule:
    def __init__(self):
        """Dummy constructor to use the Python Class as a namespace"""
        pass

    @staticmethod
    def generate_P(probs: np.ndarray, n_states: np.int32) -> np.ndarray:
        """Function that generates the probabilities (transition) matrix"""
        ### TO BE COMPLETED BY THE STUDENTS ###

        # Generate the structure of the transition matrix and intialise all to 0. T(s, a, s')
        matrixP = np.zeros((3, n_states, n_states), dtype=np.float64)

        # Loop through each state from 0 to 99.
        for s in range(n_states):
            # This action is represented as 0, 1, and 2 for decrease, maintain, and increase respectively.
            # Generate transition probabilities for action d (0).
            for i, delta in enumerate([-2, -1, 0]):
                s_result = s + delta
                prob = probs[0, i]
                # To account for out-of-bounds state less than 0
                if s_result < 0:
                    matrixP[0, s, 0] += prob
                else:
                    matrixP[0, s, s_result] += prob

            # Generate transition probabilities for action m (1).
            for i, delta in enumerate([-1, 0, 1]):
                s_result = s + delta
                prob = probs[1, i]
                # To account for out-of-bounds state less than 0
                if s_result < 0:
                    matrixP[1, s, 0] += prob
                # To account for out-of-bounds state more than 99
                elif s_result >= n_states:
                    matrixP[1, s, n_states - 1] += prob
                else:
                    matrixP[1, s, s_result] += prob

            # Generate transition probabilities for action i (2).
            for i, delta in enumerate([0, 1, 2]):
                s_result = s + delta
                prob = probs[2, i]
                # To account for out-of-bounds state more than 99
                if s_result >= n_states:
                    matrixP[2, s, n_states - 1] += prob
                else:
                    matrixP[2, s, s_result] += prob

        return matrixP

    @staticmethod
    def generate_R(demand: np.float64, n_states: np.int32) -> np.ndarray:
        """Function that generates the rewards (costs) matrix"""
        ### TO BE COMPLETED BY THE STUDENTS ###

        # Generate the structure of the reward matrix and intialise all to 0. C(s, a, s')
        matrixR = np.zeros((n_states, 3, n_states), dtype=np.float64)

        # Loop through each state from 0 to 99. Height
        for s in range(n_states):
            # Get the power of the current state level
            pow_s = s / 100.0

            # Loop through each action: 0 for decrease, 1 for maintain, and 2 for increase. Width
            for a in range(3):
                # Loop through each state from 0 to 99. Depth
                for s_result in range(n_states):
                    # Get the power of the current state level
                    pow_s_result = s_result / 100.0

                    # Calculate distance (cost) ∆t
                    distance = abs(demand - pow_s_result)
                    penalty = 1.0

                    # The distances (costs) associated with actions that move away from the target must be multiplied by ×2
                    # Current demand is greater than current state power but action is decrease (0)
                    if demand > pow_s and a == 0:
                        penalty = 2.0
                    # Current demand is less than current state power but action is increase (2)
                    elif demand < pow_s and a == 2:
                        penalty = 2.0

                    # Cost is negative reward, so store as negative value.
                    cost = penalty * distance
                    reward = -(cost)
                    matrixR[s, a, s_result] = reward

        return matrixR

    @staticmethod
    def control_iteration(
        demand: np.float64,
        state: np.int32,
        P_matrix: np.ndarray,
        gamma: np.float64,
        n_states: np.int32,
    ) -> np.int32:
        """Function that computes one control-iteration"""
        ### TO BE COMPLETED BY THE STUDENTS ###

        # Generate the reward matrix
        R_matrix = ControlModule.generate_R(demand, n_states)

        # Collapse R from (S, A, S') to (S, A) by computing expected reward
        # For each (s, a), sum over s': P(s'|s,a) * R(s,a,s')
        R_expected = np.zeros((n_states, 3), dtype=np.float64)
        for a in range(3):
            # P_matrix[a] is (S, S'), R_matrix[:,a,:] is (S, S')
            R_expected[:, a] = np.sum(P_matrix[a] * R_matrix[:, a, :], axis=1)

        # Use Value Iteration
        val_it = mdptoolbox.mdp.ValueIteration(P_matrix, R_expected, discount=gamma)

        # Run Value Iteration
        val_it.run()

        # Get the best action for the current state
        optimal_policy = val_it.policy
        best_action = optimal_policy[state]

        return best_action

    @staticmethod
    def control_loop(
        demand: np.ndarray,
        probs: np.ndarray,
        n_states: np.int32,
        n_actions: np.int32,
        gamma: np.float64,
    ) -> np.ndarray:
        """Function that computes all the required iterations (control-loop) to satisfy the power demand"""
        ### TO BE COMPLETED BY THE STUDENTS ###

        # Generate the transition matrix P
        matrixP = ControlModule.generate_P(probs, n_states)

        # Deltas for action d (0), m (1), i (2) respectively.
        delta = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]

        # Create an array to store response for demand.
        response = np.zeros_like(a=demand, dtype=np.float64)
        current_state = 0

        # Loop through each demand
        for dem in range(demand.shape[0]):
            current_demand = demand[dem]

            # Call control_iteration to find the best action
            best_action = ControlModule.control_iteration(
                demand=current_demand,
                state=current_state,
                P_matrix=matrixP,
                gamma=gamma,
                n_states=n_states,
            )

            # Randomly pick a delta
            random_delta = np.random.choice(a=delta[best_action], p=probs[best_action])

            # Get new current state
            current_state += random_delta

            # Check if out-of-bounds
            if current_state < 0:
                current_state = 0
            elif current_state >= n_states:
                current_state = n_states - 1

            # Store the response for the current demand
            response[dem] = current_state / 100.0

        return response
