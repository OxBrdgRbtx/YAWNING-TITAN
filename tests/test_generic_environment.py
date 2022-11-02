"""Used to test the GenericEnv() class and the associated agent interfaces."""
import os
import random
import warnings
from typing import Dict, List

import networkx as nx
import numpy as np
import pytest
import yaml
from stable_baselines3.common.env_checker import check_env
from yaml.loader import SafeLoader

from tests import TEST_CONFIG_PATH
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.helpers import network_creator

N_TIME_STEPS = 1000
N_TIME_STEPS_LONG = 5000


def open_config_file(settings_path: str) -> Dict:
    """
    Open a yaml environment configuration file.

    Args:
        settings_path: A path to the environment settings file

    Returns:
        settings_data: A dictionary containing the settings
    """
    with open(settings_path) as f:
        settings_data = yaml.load(f, Loader=SafeLoader)

    return settings_data


def init_test_env(
        settings_path: str, adj_matrix: np.array, positions, entry_nodes: List[str], high_value_nodes: List[str]
) -> GenericNetworkEnv:
    """
    Generate the test GenericEnv() and number of actions for the blue agent.

    Args:
        settings_path: A path to the environment settings file
        adj_matrix: the adjacency matrix used for the network to defend.
        positions: x and y co-ordinates to plot the graph in 2D space
        entry_nodes: list of strings that dictate which nodes are entry nodes
        high_value_nodes: list of strings that dictate which nodes are high value targets

    Returns:
        env: An OpenAI gym environment
    """
    if not entry_nodes:
        entry_nodes = ["0", "1", "2"]

    network_interface = NetworkInterface(
        adj_matrix, positions, entry_nodes=entry_nodes, settings_path=settings_path,
        high_value_nodes=high_value_nodes
    )

    red = RedInterface(network_interface)
    blue = BlueInterface(network_interface)

    env = GenericNetworkEnv(red, blue, network_interface)

    check_env(env, warn=True)
    env.reset()

    return env


def generate_generic_env_test_reqs(
        settings_file_path: str,
        net_creator_type="mesh",
        n_nodes: int = 10,
        connectivity: float = 0.7,
        entry_nodes=None,
        high_value_nodes=None
) -> GenericNetworkEnv:
    """
    Generate test environment requirements.

    Args:
        settings_file_path: A path to the environment settings file
        net_creator_type: The type of net creator to use to generate the underlying network
        n_nodes: The number of nodes to create within the network
        connectivity: The connectivity value for the mesh net creator (Only required for mesh network creator type)
        entry_nodes: list of strings that dictate which nodes are entry nodes
        high_value_nodes: list of strings that dictate which nodes are high value targets

    Returns:
        env: An OpenAI gym environment

    """
    valid_net_creator_types = ["18node", "mesh"]
    if net_creator_type not in valid_net_creator_types:
        raise ValueError(
            f"net_creator_type is {net_creator_type}, Must be 18_node or mesh"
        )

    if net_creator_type == "18node":
        adj_matrix, node_positions = network_creator.create_18_node_network()
    if net_creator_type == "mesh":
        adj_matrix, node_positions = network_creator.create_mesh(
            size=n_nodes, connectivity=connectivity
        )

    env = init_test_env(settings_file_path, adj_matrix, node_positions, entry_nodes, high_value_nodes)

    return env


# test 1
test_1_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "base_config.yaml"),
    net_creator_type="18node",
    n_nodes=18,
)

# test 2
test_2_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "base_config.yaml"),
    net_creator_type="mesh",
    n_nodes=50,
)

# test 3
test_3_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "base_config.yaml"),
    n_nodes=50
)

# test 4
test_4_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "red_config_test_1.yaml")
    , n_nodes=50
)

# test 5
test_5_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "red_config_test_2.yaml"),
    n_nodes=100
)

# test 6
test_6_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "red_config_test_3.yaml"),
    n_nodes=250
)
# test 7
test_7_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "red_config_test_4.yaml")
    , n_nodes=5
)

# test 8
test_8_env = generate_generic_env_test_reqs(
    os.path.join(TEST_CONFIG_PATH, "red_config_test_5.yaml"),
    n_nodes=24
)


# tests to check invalid config files return errors
def test_input_validation():
    """Test that incorrect/broken configuration files raise errors."""
    with pytest.raises(ValueError):
        _, _ = generate_generic_env_test_reqs(
            os.path.join(TEST_CONFIG_PATH, "red_config_test_broken_1.yaml"),
            n_nodes=23,
            net_creator_type="mesh",
        )

        _, _ = generate_generic_env_test_reqs(
            os.path.join(TEST_CONFIG_PATH, "red_config_test_broken_2.yaml"),
            n_nodes=23,
            net_creator_type="mesh",
        )

        _, _ = generate_generic_env_test_reqs(
            os.path.join(TEST_CONFIG_PATH, "red_config_test_broken_3.yaml"),
            n_nodes=23,
            net_creator_type="mesh",
        )

        # error thrown because choose_high_value_nodes_furthest_away_from_entry is True and
        # the high value targets is manually provided
        _, _ = generate_generic_env_test_reqs(
            os.path.join(TEST_CONFIG_PATH, "base_config.yaml"),
            n_nodes=23,
            net_creator_type="mesh",
            high_value_nodes=["0"]
        )

        # error thrown because there are more high value targets than there are nodes in the network
        _, _ = generate_generic_env_test_reqs(
            os.path.join(TEST_CONFIG_PATH, "too_many_high_value_nodes.yaml"),
            n_nodes=23,
            net_creator_type="mesh"
        )


def test_network_interface():
    """Test the network interface class and associated methods work as intended."""
    env: GenericNetworkEnv = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "everything_guaranteed.yaml"),
        net_creator_type="mesh",
        n_nodes=15,
    )
    for node in env.network_interface.get_nodes():
        assert env.network_interface.get_current_adj_matrix().all() == 0
        assert env.network_interface.get_nodes(filter_true_compromised=True) == []
        assert (
                env.network_interface.get_nodes(filter_true_safe=True)
                == env.network_interface.get_nodes()
        )
        env.network_interface.attack_node(node=node, skill=0.5, guarantee=True)
        assert env.network_interface.get_all_node_compromised_states()[node] == 1
        assert (
                sum(env.network_interface.get_all_node_compromised_states().values()) == 1
        )
        assert (
                sum(
                    env.network_interface.get_all_node_blue_view_compromised_states().values()
                )
                == 1
        )
        assert env.network_interface.get_nodes(filter_true_compromised=True) == [node]
        assert env.network_interface.get_nodes(filter_blue_view_compromised=True) == [
            node
        ]
        initial_vuln = env.network_interface.get_single_node_vulnerability(node)
        env.network_interface.update_single_node_vulnerability(node, 2)
        assert env.network_interface.get_single_node_vulnerability(node) == 2
        env.network_interface.reset_single_node_vulnerability(node)
        assert initial_vuln == env.network_interface.get_single_node_vulnerability(node)
        env.network_interface.update_stored_attacks(
            [node, node, "5"], ["2", "3", "9"], [True, False, True]
        )
        assert env.network_interface.get_true_attacks() == [
            [node, "2"],
            [node, "3"],
            ["5", "9"],
        ]

        observation_size = env.calculate_observation_space_size(with_feather=False)
        # assert the observation space is the correct size
        assert env.network_interface.get_observation_size() == observation_size

        env.network_interface.reset_stored_attacks()
        assert env.network_interface.get_true_attacks() == []
        assert env.network_interface.get_detected_attacks() == []
        env.network_interface.isolate_node(node)
        assert env.network_interface.get_nodes(filter_isolated=True) == [node]
        assert env.network_interface.get_nodes(
            filter_isolated=True, filter_true_compromised=True
        ) == [node]
        env.network_interface.reconnect_node(node)
        assert env.network_interface.get_nodes(filter_isolated=True) == []
        env.network_interface.make_node_safe(node)
        assert env.network_interface.get_nodes(filter_true_compromised=True) == []
        assert env.network_interface.get_nodes(filter_blue_view_compromised=True) == []
        env.reset()


def test_natural_spreading():
    """Test the natural spreading simulation mechanic works as intended."""
    # generate an env
    n_nodes = 100
    env = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "spreading_config.yaml"),
        net_creator_type="mesh",
        n_nodes=n_nodes,
    )
    check_env(env, warn=True)
    env.reset()
    total_cum_success = 0
    for i in range(N_TIME_STEPS):
        # step through the environment and count the number of attacks, red cannot perform any actions so the only attacks are from natural spreading
        env.step(0)
        total_cum_success += len(env.network_interface.get_true_attacks()) / n_nodes
    spreading_success_rate = total_cum_success / N_TIME_STEPS
    # ensure that the number of spreads is within a reasonable degree of accuracy of the set spreading rate
    assert 0.0185 < spreading_success_rate < 0.0215


# check to make sure than when an env is reset all of the proper values are reset too
def test_env_reset():
    """Test environment resets clean up properly."""
    env = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "base_config.yaml"),
        net_creator_type="mesh",
        n_nodes=15,
    )

    check_env(env, warn=True)
    env.reset()
    for i in range(0, N_TIME_STEPS):
        obs, rew, done, notes = env.step(
            random.randint(0, env.BLUE.get_number_of_actions() - 1)
        )
        if done:
            env.reset()
            # check no deceptive nodes left in the environment
            assert not any(
                node.startswith("d")
                for node in env.network_interface.current_network_variables.keys()
            )

            # check red agent is reset
            assert env.network_interface.red_current_location is None
            # check all nodes are set back to their initial state
            assert (
                    1
                    not in env.network_interface.get_all_node_blue_view_compromised_states().values()
            )
            assert (
                    1
                    not in env.network_interface.get_all_node_compromised_states().values()
            )
            assert env.network_interface.get_nodes(filter_true_compromised=True) == []
            # check deceptive node pointers are reset
            assert env.network_interface.deceptive_node_pointer == 0
            assert env.network_interface.current_deceptive_nodes == 0
            assert env.network_interface.reached_max_deceptive_nodes is False
            # check the network is the same as the base
            assert nx.is_isomorphic(
                env.network_interface.base_graph, env.network_interface.current_graph
            )
            # check all previous attacks are removed
            assert env.network_interface.true_attacks == []
            assert env.network_interface.detected_attacks == []


def test_new_high_value_node():
    """Test the high value target gaol mechanic - focus on selection."""
    # check that a new high value target is being chosen
    env = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "new_high_value_target.yaml"),
        net_creator_type="mesh",
        n_nodes=15,
    )
    check_env(env, warn=True)
    env.reset()
    targets = {}
    for i in range(0, N_TIME_STEPS_LONG):
        obs, rew, done, notes = env.step(
            random.randint(0, env.BLUE.get_number_of_actions() - 1)
        )
        if done:
            hvn = env.network_interface.get_high_value_nodes()

            # add 1 to each node that gets chosen as a high value target
            for node in hvn:
                if node not in targets:
                    targets[node] = 1
                else:
                    targets[node] += 1

            env.reset()
    # check that entry nodes cannot be chosen
    # 3 entry nodes are configured in new_high_value_target.yaml, so n_nodes - number_of_entry_nodes = 12
    assert len(targets.keys()) == 12
    # check that each node is roughly chosen equally
    sum_ = sum(targets.values())
    for i in targets.values():
        assert 1 / 10.5 > i / sum_ > 1 / 13.5


def test_high_value_target_passed_into_network_interface():
    """Test the high value target gaol mechanic - manually passed to ."""
    env = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "high_value_target_provided.yaml"),
        net_creator_type="mesh",
        n_nodes=30,
        high_value_nodes=["15", "16"]
    )
    check_env(env, warn=True)
    env.reset()
    targets = {}
    for i in range(0, N_TIME_STEPS_LONG):
        obs, rew, done, notes = env.step(
            random.randint(0, env.BLUE.get_number_of_actions() - 1)
        )
        if done:
            hvn = env.network_interface.get_high_value_nodes()

            # add 1 to each node that gets chosen as a high value target
            for node in hvn:
                if node not in targets:
                    targets[node] = 1
                else:
                    targets[node] += 1

            env.reset()
    # the only 2 targets set are available
    assert len(targets.keys()) == 2

    # check that the keys are the 15 and 16 that was set
    assert set(targets.keys()).intersection(["15", "16"])


def test_high_value_target_and_entry_nodes_matching():
    """Test the high value target gaol mechanic - manually passed to ."""
    with warnings.catch_warnings(record=True) as w:
        env = generate_generic_env_test_reqs(
            os.path.join(TEST_CONFIG_PATH, "high_value_target_provided.yaml"),
            net_creator_type="mesh",
            n_nodes=30,
            entry_nodes=["0", "15"],
            high_value_nodes=["15", "16"]
        )
        check_env(env, warn=True)
        env.reset()

        entry = {}
        targets = {}
        for i in range(0, N_TIME_STEPS_LONG):
            obs, rew, done, notes = env.step(
                random.randint(0, env.BLUE.get_number_of_actions() - 1)
            )
            if done:
                hvn = env.network_interface.get_high_value_nodes()

                # add 1 to each node that gets chosen as a high value target
                for node in hvn:
                    if node not in targets:
                        targets[node] = 1
                    else:
                        targets[node] += 1

                env.reset()
        # the only 2 targets set are available
        assert len(targets.keys()) == 2

        # check that the keys are the 15 and 16 that was set
        assert set(targets.keys()).intersection(["15", "16"])

        # check that a warning was raised that the entry nodes and high value targets intersect
        assert str(w[0].message.args[0]) == "Provided entry nodes and high value targets intersect and may cause the training to prematurely end"


def test_new_entry_nodes():
    """Test the selection of entry nodes and validate they are correct."""
    env = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "new_entry_nodes.yaml"),
        net_creator_type="mesh",
        n_nodes=15,
    )
    check_env(env, warn=True)
    env.reset()
    entry_nodes = {}
    for i in range(0, N_TIME_STEPS_LONG):
        obs, rew, done, notes = env.step(
            random.randint(0, env.BLUE.get_number_of_actions() - 1)
        )
        if done:
            entry = env.network_interface.get_entry_nodes()
            for node in entry:
                if node not in entry_nodes:
                    entry_nodes[node] = 1
                else:
                    entry_nodes[node] += 1
            env.reset()
    # check that entry nodes cannot be chosen
    assert len(entry_nodes.keys()) == 15
    # check that each node is roughly chosen equally
    for i in entry_nodes.values():
        assert np.isclose(i, 1000, atol=100)


def test_new_vulnerabilities():
    """Test that new vulnerabilities are chosen at each reset if activated within configuration."""
    # check that new vulnerabilities are being chosen (randomly)
    env = generate_generic_env_test_reqs(
        os.path.join(TEST_CONFIG_PATH, "new_high_value_target.yaml"),
        net_creator_type="mesh",
        n_nodes=15,
    )
    check_env(env, warn=True)
    env.reset()
    vulnerabilities = 0
    resets = 0
    for i in range(0, N_TIME_STEPS):
        obs, rew, done, notes = env.step(
            random.randint(0, env.BLUE.get_number_of_actions() - 1)
        )
        if done:
            vulnerabilities += sum(
                env.network_interface.get_all_vulnerabilities().values()
            )
            resets += 1
            env.reset()
    # calculate the average vulnerability
    vulnerabilities = (vulnerabilities / 15) / resets
    vuln_aim = env.network_interface.gr_node_vuln_lower + 0.5 * (
            env.network_interface.gr_node_vuln_upper
            - env.network_interface.gr_node_vuln_lower
    )
    # ensure the average vulnerability is half way between the upper and lower range
    assert (vuln_aim - 0.01 * vuln_aim) < vulnerabilities < (vuln_aim + 0.01 * vuln_aim)


class RandomGen:
    """Used to ensure that every action is chosen but also that there are random actions chosen."""

    def __init__(self, number_actions):
        self.number_of_actions = number_actions
        self.possible_actions = [i for i in range(0, number_actions)]
        self.test_repeat = False
        self.amount = 5
        self.repeat_action = 0

    def get_action(self):
        """Return an action."""
        if not self.test_repeat:
            if random.randint(0, 20) > 19:
                self.test_repeat = True
                self.repeat_action = random.randint(0, self.number_of_actions)

        if self.test_repeat:
            self.amount -= 1
            if self.amount <= 0:
                self.test_repeat = False
                self.amount = 5
            return self.repeat_action

        chosen_action = random.choice(self.possible_actions)
        self.possible_actions.remove(chosen_action)
        if len(self.possible_actions) == 0:
            self.possible_actions = [i for i in range(0, self.number_of_actions)]
        return chosen_action


@pytest.mark.parametrize(
    ("env", "timesteps"),
    [
        (test_1_env, N_TIME_STEPS),
        (test_2_env, N_TIME_STEPS),
        (test_3_env, N_TIME_STEPS),
        (test_4_env, N_TIME_STEPS),
        (test_5_env, N_TIME_STEPS),
        (test_6_env, N_TIME_STEPS),
        (test_7_env, N_TIME_STEPS),
        (test_8_env, N_TIME_STEPS),
    ],
)
def test_generic_env(env, timesteps):
    """
    Test the generic environment across a number of test environments.

    Args:
        env: The GenerciEnv() OpenAI gym used for as the test environment
        timesteps: The number of timesteps to test for
    """
    # Counters for later
    discovered_from_scanning = 0
    nodes_missed_scan = 0
    turn_counter = 0
    discovered_immediately = 0
    number_of_resets = 0
    wins = 0
    deceptive_counter = 0

    red_action_count = {True: {}, False: {}}
    multi_attack_count = {True: {}, False: {}}
    blue_action_count = {}
    initial_edges = env.network_interface.get_number_base_edges()

    env.reset()

    random_action_generator = RandomGen(env.BLUE.get_number_of_actions())

    prev_high_value = None

    for t_step in range(0, timesteps):
        current_chosen_action = random_action_generator.get_action()
        obs, rew, done, notes = env.step(current_chosen_action)

        turn_counter += 1
        blue_action = notes["blue_action"]
        blue_node = notes["blue_node"]
        initial_state = notes["initial_state"]
        post_red_state = notes["post_red_state"]
        end_state = notes["end_state"]
        initial_blue_view = notes["initial_blue_view"]
        post_red_blue_view = notes["post_red_blue_view"]
        end_blue_view = notes["end_blue_view"]
        initial_vulnerabilities = notes["initial_vulnerabilities"]
        post_red_vulnerabilities = notes["post_red_vulnerabilities"]
        final_vulnerabilities = notes["final_vulnerabilities"]
        initial_red_location = notes["initial_red_location"]
        post_red_red_location = notes["post_red_red_location"]
        final_red_location = notes["final_red_location"]
        attacks = notes["attacks"]
        initial_graph = notes["initial_graph"]
        red_info = notes["red_info"]
        # natural spreading
        ns = list(filter(lambda x: x["Action"] == "natural_spread", red_info.values()))

        assert len(ns) == 1 or len(ns) == 0
        if len(ns) == 1:
            ns = ns[0]
            for i in [
                index
                for index, attack_success in enumerate(ns["Successes"])
                if attack_success
            ]:
                assert post_red_state[ns["Target_Nodes"][i]] == 1
        # random high value target placement
        if not done:
            if not env.network_interface.gr_loss_hvn_random_placement:
                assert (
                        env.network_interface.get_high_value_nodes() == prev_high_value
                        or prev_high_value is None
                )

        assert (
                env.network_interface.adj_matrix.all()
                == nx.to_numpy_array(env.network_interface.current_graph).all()
        )

        # making sure that attacks come from a connected node
        prev_attacks = []
        for i in attacks:
            assert i[0] is None or initial_state[i[0]] == 1 or i[0] in prev_attacks
            assert initial_state[i[1]] == 0
            prev_attacks.append(i[1])
            if blue_action != "isolate" and blue_action != "add_deceptive_node":
                if i[0] not in env.network_interface.get_current_connected_nodes(
                        i[1]
                ) + [None]:
                    raise AssertionError(
                        i,
                        env.network_interface.get_current_connected_nodes(i[0]),
                        blue_action,
                        blue_node,
                        red_info,
                    )
            elif blue_action == "isolate" and i[1] == blue_node:
                if i[0] is not None:
                    assert i[0] in list(nx.Graph(initial_graph).neighbors(blue_node))
            elif blue_action == "add_deceptive_node" and i[1] == blue_node[0]:
                if blue_node in nx.Graph(initial_graph):
                    assert i[0] in list(nx.Graph(initial_graph).neighbors(blue_node[0]))

        # only use the set number of deceptive nodes
        assert (
                sum("d" in s for s in env.network_interface.get_nodes())
                <= env.network_interface.blue_max_deceptive_nodes
        )
        # cannot place too many deceptive nodes
        assert (
                initial_edges
                <= env.network_interface.get_number_base_edges()
                <= (initial_edges + env.network_interface.blue_max_deceptive_nodes)
        )

        # observation space is correct
        adj_matrix = nx.to_numpy_array(env.network_interface.current_graph)
        open_spaces = env.network_interface.get_number_unused_deceptive_nodes()
        adj_matrix = np.pad(adj_matrix, (0, open_spaces), "constant").flatten()
        if env.network_interface.obs_node_connections:
            assert np.array_equal(obs[: len(adj_matrix)], adj_matrix)

        observation_size = env.calculate_observation_space_size(with_feather=False)

        # assert the observation space is the correct size
        assert env.network_interface.get_observation_size() == observation_size

        if current_chosen_action > env.action_space.n:
            assert notes["blue_action"] == "do_nothing"

        # red location (if using the red can only attack from central red node)
        if initial_red_location is None:
            assert (
                    post_red_red_location in env.network_interface.get_entry_nodes()
                    or post_red_red_location is None
            )
        elif post_red_red_location is None:
            assert final_red_location is None
        else:
            if blue_action != "add_deceptive_node":
                connected = env.network_interface.get_current_connected_nodes(
                    post_red_red_location
                )
                connected = [
                    i
                    for i in connected
                    if env.network_interface.get_single_node_state(i) == 1
                ]
                if len(connected) == 0:
                    assert final_red_location == post_red_red_location or (
                            (
                                    blue_action == "restore_node"
                                    or blue_action == "make_node_safe"
                            )
                            and blue_node == post_red_red_location
                            and final_red_location is None
                    )
                else:
                    if blue_action == "restore_node" or blue_action == "make_node_safe":
                        if blue_node == post_red_red_location:
                            assert final_red_location in connected
                    else:
                        assert post_red_red_location == final_red_location
        if initial_red_location is None:
            connected = env.network_interface.get_entry_nodes()
        else:
            connected = env.network_interface.get_current_connected_nodes(
                initial_red_location
            )
        if blue_action != "isolate" or (
                blue_node != post_red_red_location and blue_node != initial_red_location
        ):
            if blue_action != "add_deceptive_node":
                assert (
                        post_red_red_location in connected
                        or post_red_red_location == initial_red_location
                )

        assert 0 in initial_state.values()

        # used to calculate number of successes and failures for different actions
        for action in red_info.values():
            red_action = action["Action"]
            red_success = action["Successes"]
            red_target = action["Target_Nodes"]
            if red_action != "natural_spread" and red_action != "no_possible_targets":
                if red_action != "spread" and red_action != "intrude":
                    # if a action that targets a single node succeeds or fails
                    for i in red_success:
                        if red_action not in red_action_count[i]:
                            red_action_count[i][red_action] = 1
                        else:
                            red_action_count[i][red_action] += 1
                else:
                    # multi-node target
                    overall_success = True in red_success

                    # adds one attack action to the red action count (this is so actions can easily be counted to max timesteps)
                    if red_action not in red_action_count[overall_success]:
                        red_action_count[overall_success][red_action] = 1
                    else:
                        red_action_count[overall_success][red_action] += 1

                    # counts all the individual successes and failures against the nodes
                    for success in red_success:
                        if red_action not in multi_attack_count[success]:
                            multi_attack_count[success][red_action] = 1
                        else:
                            multi_attack_count[success][red_action] += 1

        if blue_action not in blue_action_count:
            blue_action_count[blue_action] = 1
        else:
            blue_action_count[blue_action] += 1

        # RED ACTIONS

        for action in red_info.values():
            red_action = action["Action"]
            red_success = action["Successes"]
            red_target = action["Target_Nodes"]
            red_attacking_node = action["Attacking_Nodes"]
            if red_action == "basic_attack":
                for counter, current_success in enumerate(red_success):
                    if (
                            env.network_interface.red_chance_to_spread_to_unconnected_node
                            == 0
                    ):
                        assert (
                            not env.network_interface.get_single_node_isolation_status(
                                red_target[counter]
                            )
                        )
                    if current_success:
                        if blue_action != "isolate" or (
                                blue_action == "isolate"
                                and blue_node != red_attacking_node[counter]
                                and blue_node != red_target[counter]
                        ):
                            if blue_action != "add_deceptive_node" or (
                                    blue_action == "add_deceptive_node"
                                    and red_target[counter] != blue_node[0]
                                    and red_attacking_node[counter] != blue_node[0]
                                    and not (
                                    red_target[counter] in blue_node[1]
                                    and red_attacking_node[counter] in blue_node[1]
                            )
                            ):
                                if (
                                        red_target[counter]
                                        not in env.network_interface.get_entry_nodes()
                                ):
                                    assert red_target[
                                               counter
                                           ] in env.network_interface.get_current_connected_nodes(
                                        red_attacking_node[counter]
                                    )
                        assert post_red_state[red_target[counter]] == 1
                        if post_red_blue_view[red_target[counter]] == 1:
                            discovered_immediately += 1
                        if (
                                red_target[counter][0] == "d"
                                and env.network_interface.blue_deception_immediate_detection_chance
                                == 1
                        ):
                            assert post_red_blue_view[red_target[counter]] == 1
                        if (
                                env.network_interface.blue_immediate_detection_chance == 0
                                and red_target[counter][0] != "d"
                        ):
                            assert post_red_blue_view[red_target[counter]] == 0
                        if (
                                env.network_interface.blue_immediate_detection_chance == 1
                                and red_target[counter][0] != "d"
                        ):
                            assert post_red_blue_view[red_target[counter]] == 1
            elif red_action == "zero_day":
                for i in red_success:
                    assert i
                for i in red_target:
                    assert post_red_state[i] == 1
                    if post_red_blue_view[i] == 1:
                        discovered_immediately += 1
            elif red_action == "spread":
                for counter, success in enumerate(red_success):
                    if success:
                        assert post_red_state[red_target[counter]] == 1
                        if post_red_blue_view[red_target[counter]] == 1:
                            discovered_immediately += 1
            elif red_action == "intrude":
                for counter, success in enumerate(red_success):
                    if success:
                        assert post_red_state[red_target[counter]] == 1
                        if post_red_blue_view[red_target[counter]] == 1:
                            discovered_immediately += 1
            elif red_action == "random_move":
                for counter, current_success in enumerate(red_success):
                    if current_success:
                        if blue_action != "isolate" or (
                                blue_action == "isolate"
                                and blue_node != red_attacking_node[counter]
                                and blue_node != red_target[counter]
                        ):
                            if blue_action != "add_deceptive_node" or (
                                    blue_action == "add_deceptive_node"
                                    and red_target[counter] != blue_node[0]
                                    and red_attacking_node[counter] != blue_node[0]
                                    and not (
                                    red_target[counter] in blue_node[1]
                                    and red_attacking_node[counter] in blue_node[1]
                            )
                            ):
                                if red_attacking_node[counter] is None:
                                    assert (
                                            red_target[counter]
                                            in env.network_interface.get_entry_nodes()
                                    )
                                else:
                                    assert red_target[
                                               counter
                                           ] in env.network_interface.get_current_connected_nodes(
                                        red_attacking_node[counter]
                                    )
            elif red_action == "do_nothing":
                assert initial_red_location == post_red_red_location
                assert initial_vulnerabilities == post_red_vulnerabilities
                if len(red_info) == 1:
                    assert initial_state == post_red_state
                    assert initial_blue_view == post_red_blue_view
            elif red_action == "no_possible_targets":
                if (
                        env.network_interface.red_attack_from_any_node
                        and not env.network_interface.blue_isolate_action
                ):
                    assert (
                            len(env.network_interface.get_nodes(filter_true_safe=True)) == 0
                    )
                assert initial_vulnerabilities == post_red_vulnerabilities
            elif red_action == "natural_spread":
                pass
            else:
                raise AssertionError("Missing red action:" + red_action)

        # Fix up
        # if True not in natural_spreading_success:
        #    assert initial_state == post_red_state
        #    assert initial_blue_view == post_red_blue_view

        # BLUE ACTIONS
        scan_used = False
        if blue_action == "restore_node" or blue_action == "make_node_safe":
            assert end_state[blue_node] != 1
            if "d" in blue_node:
                deceptive_counter += 1
        elif blue_action == "isolate":
            assert (
                    len(env.network_interface.get_current_connected_nodes(blue_node)) == 0
            )
            assert env.network_interface.get_single_node_isolation_status(blue_node)
        elif blue_action == "connect":
            for i in env.network_interface.get_current_connected_nodes(blue_node):
                assert not env.network_interface.get_single_node_isolation_status(i)
            for i in env.network_interface.get_base_connected_nodes(blue_node):
                if not env.network_interface.get_single_node_isolation_status(i):
                    assert i in env.network_interface.get_current_connected_nodes(
                        blue_node
                    )
            assert not env.network_interface.get_single_node_isolation_status(blue_node)
        elif blue_action == "reduce_vulnerability":
            assert 0 < final_vulnerabilities[blue_node] < 1
            assert (
                    final_vulnerabilities[blue_node] <= post_red_vulnerabilities[blue_node]
            )

        elif blue_action == "scan":
            scan_used = True
            for node in env.network_interface.get_nodes():
                if initial_blue_view[node] == 1:
                    assert end_blue_view[node] == 1

                if post_red_blue_view[node] == 0 and end_blue_view[node] == 1:
                    discovered_from_scanning += 1

                if end_state[node] != end_blue_view[node]:
                    nodes_missed_scan += 1
                if "d" not in node:
                    if env.network_interface.blue_scan_detection_chance == 1:
                        if end_state[node] == 1:
                            assert end_blue_view[node] == 1
                else:
                    if env.network_interface.blue_deception_scan_detection_chance == 1:
                        if end_state[node] == 1:
                            assert end_blue_view[node] == 1

        elif blue_action == "add_deceptive_node":
            deceptive_node = blue_node[0]
            node1 = blue_node[1][0]
            node2 = blue_node[1][1]
            assert node2 not in env.network_interface.get_base_connected_nodes(node1)
            assert [
                       i
                       for i in env.network_interface.get_base_connected_nodes(deceptive_node)
                       if i != node1 and i != node2
                   ] == []
            for i in env.network_interface.get_nodes():
                if i != node1 and i != node2:
                    assert (
                            deceptive_node
                            not in env.network_interface.get_base_connected_nodes(i)
                    )
            assert env.network_interface.current_deceptive_nodes != 0
            assert (
                    env.network_interface.get_single_node_vulnerability(deceptive_node)
                    is not None
            )
            assert (
                    env.network_interface.get_single_node_state(deceptive_node) is not None
            )

            if deceptive_node in post_red_state:
                if env.network_interface.blue_deceptive_node_make_new:
                    if post_red_state[deceptive_node] == 1:
                        assert end_state[deceptive_node] == 0
                        assert (
                                env.network_interface.get_single_node_isolation_status(
                                    deceptive_node
                                )
                                == 0
                        )
                else:
                    assert end_state[deceptive_node] == post_red_state[deceptive_node]
                    assert (
                            final_vulnerabilities[deceptive_node]
                            == post_red_vulnerabilities[deceptive_node]
                    )
                    assert (
                            end_blue_view[deceptive_node]
                            == post_red_blue_view[deceptive_node]
                    )
            else:
                assert end_state[deceptive_node] == 0
                assert (
                        env.network_interface.get_single_node_isolation_status(
                            deceptive_node
                        )
                        == 0
                )
            assert env.network_interface.get_single_node_position(deceptive_node) != 0

        elif blue_action == "do_nothing" or blue_action == "failed":
            assert post_red_state == end_state
            assert post_red_blue_view == end_blue_view
            assert post_red_vulnerabilities == final_vulnerabilities
            assert post_red_red_location == final_red_location
        else:
            raise AssertionError("Missing blue action:" + blue_action)

        if done and not blue_action == "failed":
            wins += 1

        if done is True:
            if (
                    env.current_duration > 500
                    and env.network_interface.get_number_of_nodes() < 15
            ):
                if (
                        env.network_interface.blue_make_node_safe_action
                        or env.network_interface.blue_restore_node_action
                ):
                    if "d0" in env.network_interface.get_nodes():
                        assert deceptive_counter != 0
            if env.current_duration != env.network_interface.gr_max_steps:
                if env.network_interface.gr_loss_total_compromise:
                    assert (
                            len(env.network_interface.get_nodes(filter_true_safe=True)) == 0
                    )
                if env.network_interface.gr_loss_pc_nodes_compromised:
                    assert (
                                   len(
                                       env.network_interface.get_nodes(
                                           filter_true_compromised=True
                                       )
                                   )
                                   / len(env.network_interface.get_nodes())
                           ) >= env.network_interface.gr_loss_pc_node_compromised_pc
                if env.network_interface.gr_loss_hvn:
                    # the game ends when a high value node is compromised, this needs to be checked
                    compromised_hvn = False

                    for node in env.network_interface.get_high_value_nodes():
                        if env.network_interface.get_single_node_state(node) == 1:
                            compromised_hvn = True
                            break

                    assert (compromised_hvn is True)
            env.reset()
            number_of_resets += 1
        else:
            if env.network_interface.gr_loss_total_compromise:
                assert len(env.network_interface.get_nodes(filter_true_safe=True)) != 0
            if env.network_interface.gr_loss_pc_nodes_compromised:
                assert (
                               len(env.network_interface.get_nodes(filter_true_compromised=True))
                               / len(env.network_interface.get_nodes())
                       ) < env.network_interface.gr_loss_pc_node_compromised_pc
            if env.network_interface.gr_loss_hvn:
                # the game would end if a high value target was compromised, this needs to be checked
                compromised_hvn = False

                # check that none of the high value targets are compromised
                for node in env.network_interface.get_high_value_nodes():
                    # get_single_node_state returns 1 if compromised
                    if env.network_interface.get_single_node_state(node) == 1:
                        compromised_hvn = True

                assert (compromised_hvn is False)

    # tests on data collected from trial
    if not env.network_interface.blue_deceptive_action and scan_used:
        assert (
                (0.95 * env.network_interface.blue_scan_detection_chance)
                < (
                        discovered_from_scanning
                        / (nodes_missed_scan + discovered_from_scanning)
                )
                < (1.05 * env.network_interface.blue_scan_detection_chance)
        )

    assert (
            sum(red_action_count[True].values()) + sum(red_action_count[False].values())
            == timesteps
    )
    assert sum(blue_action_count.values()) == timesteps

    if "failed" in blue_action_count:
        assert number_of_resets == blue_action_count["failed"] + wins
    else:
        assert wins == number_of_resets

    env.close()
