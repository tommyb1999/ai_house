import copy
import pprint
import numpy as np
from typing import Tuple, List, Dict, Optional

from gaz_singleplayer.config_syngame import Config
from environment.flowsheet_simulation import FlowsheetSimulation
from environment.env_config import EnvConfig

class SinglePlayerGame:
    """
    Class representing a single-player task for flowsheet synthesis.
    """
    
    def __init__(self, config: Config, flowsheet_simulation_config: EnvConfig, problem_instance: List):
        """
        Initialize the game.

        Parameters
        ----------
        config : Config
            Configuration for the game.
        flowsheet_simulation_config : EnvConfig
            Environment-specific parameters and settings.
        problem_instance : List
            List of feed streams representing the problem instance.
        """
        self.config = config
        self.flowsheet_simulation_config = flowsheet_simulation_config
        self.problem_instance = problem_instance
        self.level_current_player = 0
        self.action_current_player = {
            "line_index": None,
            "unit_index": None,
            "spec_cont": None,
            "spec_disc": None
        }
        self.game_broken = False
        
        self.player_environment = FlowsheetSimulation(
            copy.deepcopy(self.problem_instance), self.flowsheet_simulation_config
        )
        
        self.current_feasible_actions = self.player_environment.get_feasible_actions(
            current_level=self.level_current_player,
            chosen_stream=self.action_current_player["line_index"],
            chosen_unit=self.action_current_player["unit_index"]
        )
        
        self.game_is_over = False
        self.player_npv = -1 * float("inf")
        self.player_npv_explicit = -1 * float("inf")

    def get_current_level(self):
        return self.level_current_player

    def get_objective(self, for_player: int) -> float:
        return self.player_npv

    def get_explicit_npv(self, for_player: int) -> float:
        return self.player_npv_explicit

    def get_sequence(self, for_player: int) -> Dict:
        return self.player_environment.blueprint

    def get_number_of_lvl_zero_moves(self) -> int:
        return self.player_environment.steps

    def get_num_actions(self) -> int:
        """Legal actions for the current player at the current level given as a list of ints."""
        return len(self.current_feasible_actions)

    def get_feasible_actions_ohe_vector(self) -> np.array:
        return self.current_feasible_actions

    def is_finished_and_winner(self) -> Tuple[bool, int]:
        return self.game_is_over, 0  # Irrelevant for single-player games

    def make_move(self, action: int) -> Tuple[bool, float, bool]:
        """
        Performs a move in the game environment, in the flowsheet case this does not necessarily mean 
        that a unit is placed as the action may not be complete.
        """
        if self.game_broken:
            raise Exception('Playing in a broken game')
        
        if self.current_feasible_actions[action] != 1:
            raise Exception("Playing infeasible action.")
        
        # Action selection logic based on current level
        if self.level_current_player == 0:
            self.action_current_player["line_index"] = action
        elif self.level_current_player == 1:
            self.action_current_player["unit_index"] = action
        elif self.level_current_player == 2:
            self.action_current_player["spec_disc"] = action
        elif self.level_current_player == 3:
            self.action_current_player["spec_cont"] = [None, [action, None]]
        
        # Set next level
        next_level = self._determine_next_level(action)
        
        # Execute action in the simulation if next_level is 0
        game_done, reward, move_worked = self._execute_action_if_final(next_level)
        
        if not self.game_is_over:
            self.level_current_player = next_level
            if next_level == 0:
                self.action_current_player = {"line_index": None, "unit_index": None, "spec_cont": None, "spec_disc": None}
            
            if not self.game_broken:
                self.current_feasible_actions = self.player_environment.get_feasible_actions(
                    current_level=self.level_current_player,
                    chosen_stream=self.action_current_player["line_index"],
                    chosen_unit=self.action_current_player["unit_index"]
                )
        
        return game_done, reward, move_worked

    def get_current_state(self):
        """Returns the current situation as a dictionary."""
        if self.game_broken:
            raise Exception("Getting state of a broken game.")
        
        dict_current_state = {
            "current_npv": self.player_environment.current_net_present_value * self.config.objective_scaling,
            "num_lines": len(self.player_environment.state_simulation["list_line_information"]),
            "action_level": self.level_current_player,
            "feasible_actions": self.current_feasible_actions,
            "flowsheet_finished": self.player_environment.state_simulation["flowsheet_syn_done"],
            "chosen_stream": self.action_current_player["line_index"],
        }
        return dict_current_state

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def generate_random_instance(flowsheet_simulation_config) -> Dict:
        return flowsheet_simulation_config.create_random_problem_instance()

# Main Function
if __name__ == "__main__":
    config = Config()
    env_config = EnvConfig()
    problem_instance = env_config.create_random_problem_instance()
    
    game = SinglePlayerGame(config, env_config, problem_instance)
    print("Game initialized.")