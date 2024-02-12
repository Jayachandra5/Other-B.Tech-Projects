from tkinter import *
import numpy as np
from numpy import ndarray
from typing import NamedTuple, Literal, Tuple
from time import time


# Constants for styling
size_of_board = 600
number_of_dots = 4
symbol_size = (size_of_board / 3 - size_of_board / 8) / 2
symbol_thickness = 50
dot_color = "black"
player1_color = "#0492CF"
player1_color_light = "#67B0CF"
player2_color = "#EE4035"
player2_color_light = "#EE7E77"
Green_color = "#7BC043"
dot_width = 0.1 * size_of_board / number_of_dots
edge_width = 0.1 * size_of_board / number_of_dots
distance_between_dots = size_of_board / (number_of_dots)

BOT_TURN_INTERVAL_MS = 100
LEFT_CLICK = "<Button-1>"
TIMEOUT = 5


# Data structures for game actions and state
class Action(NamedTuple):
    action_type: Literal["row", "col"]
    position: Tuple[int, int]


class States(NamedTuple):
    board_status: ndarray
    row_status: ndarray
    col_status: ndarray
    player1_turn: bool


# Ai agent
class Bot:
    def __init__(self):
        self.is_player1 = True
        self.global_time = 0

    def best_decision(self, state: States) -> Action:
        """
        Get the best action for the current game state using adversarial search.

        Args:
            state (States): Current game state.

        Returns:
            Action: Best action to be taken.
        """
        self.is_player1 = state.player1_turn
        selected_action: Action = None
        self.global_time = time() + TIMEOUT

        row_not_filled = np.count_nonzero(state.row_status == 0)
        column_not_filled = np.count_nonzero(state.col_status == 0)

        for i in range(row_not_filled + column_not_filled):
            try:
                actions = self.create_required_move(state)
                utilities = np.array([self.minimaxAlphaBeta(
                    state=self.results(state, action),
                    max_depth=i + 1) for action in actions])
                index = np.random.choice(
                    np.flatnonzero(utilities == utilities.max()))
                selected_action = actions[index]
            except TimeoutError:
                break

        return selected_action

    def create_required_move(self, state):
        """
        Generate possible actions for the given game state.

        Args:
            state (States): Current game state.

        Returns:
            List[Action]: List of possible actions.
        """
        row_positions = self.valid_moves(state.row_status)
        col_positions = self.valid_moves(state.col_status)
        actions = []
        for position in row_positions:
            actions.append(Action("row", position))
        for position in col_positions:
            actions.append(Action("col", position))

        return actions

    def valid_moves(self, matrix: np.ndarray):
        """
        Generate valid positions for the given matrix.

        Args:
            matrix (np.ndarray): Matrix to generate positions from.

        Returns:
            List[Tuple[int, int]]: List of valid positions.
        """
        [ny, nx] = matrix.shape
        positions = []

        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))

        return positions

    def results(self, state: States, action: Action) -> States:
        """
        Get the game state resulting from taking a specific action.

        Args:
            state (States): Current game state.
            action (Action): Action to be taken.

        Returns:
            States: Resulting game state.
        """
        type = action.action_type
        x, y = action.position

        new_state = States(
            state.board_status.copy(),
            state.row_status.copy(),
            state.col_status.copy(),
            state.player1_turn,
        )

        player_modifier = -1 if new_state.player1_turn else 1

        is_point_scored = False
        val = 1

        [ny, nx] = new_state.board_status.shape

        if y < ny and x < nx:
            new_state.board_status[y, x] = (
                abs(new_state.board_status[y, x]) + val
            ) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True

        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (
                    abs(new_state.board_status[y - 1, x]) + val
                ) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True

        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (
                    abs(new_state.board_status[y, x - 1]) + val
                ) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True

        new_state = new_state._replace(
            player1_turn=not (new_state.player1_turn ^ is_point_scored)
        )

        return new_state

    def minimaxAlphaBeta(
        self,
        state: States,
        depth: int = 0,
        max_depth: int = 0,
        alpha: float = -np.inf,
        beta: float = np.inf,
    ) -> float:
        """
        Get the minimax value for the given state and depth.

        Args:
            state (States): Current game state.
            depth (int): Current depth in the search tree.
            max_depth (int): Maximum depth to explore.
            alpha (float): Alpha value for alpha-beta pruning.
            beta (float): Beta value for alpha-beta pruning.

        Returns:
            float: Minimax value for the state.
        """
        if time() >= self.global_time:
            raise TimeoutError()

        if self.check(state) or depth == max_depth:
            return self.calc_util_value(state)

        if self.is_player1 == state.player1_turn:
            value = -np.inf
            actions = self.create_required_move(state)
            for action in actions:
                value = max(
                    value,
                    self.minimaxAlphaBeta(
                        self.results(state, action),
                        depth=depth + 1,
                        max_depth=max_depth,
                        alpha=alpha,
                        beta=beta
                    ),
                )
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = np.inf
            actions = self.create_required_move(state)
            for action in actions:
                value = min(
                    value,
                    self.minimaxAlphaBeta(
                        self.results(state, action),
                        depth=depth + 1,
                        max_depth=max_depth,
                        alpha=alpha,
                        beta=beta
                    ),
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def check(self, state: States) -> bool:
        """
        Check if the given state is a terminal state.

        Args:
            state (States): Current game state.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return np.all(state.row_status == 1) and np.all(state.col_status == 1)

    def calc_util_value(self, state: States) -> float:
        """
        Calculate the utility value for the given state.

        Args:
            state (States): Current game state.

        Returns:
            float: Utility value for the state.
        """
        [ny, nx] = state.board_status.shape
        utility = 0

        box_won = 0
        box_lost = 0
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if state.board_status[y, x] == -4:
                        utility += 1
                        box_won += 1
                    elif state.board_status[y, x] == 4:
                        utility -= 1
                        box_lost += 1
                else:
                    if state.board_status[y, x] == -4:
                        utility -= 1
                        box_lost += 1
                    elif state.board_status[y, x] == 4:
                        utility += 1
                        box_won += 1

        if self.count_chains(state) % 2 == 0 and self.is_player1:
            utility += 1
        elif self.count_chains(state) % 2 != 0 and not self.is_player1:
            utility += 1

        if box_won >= 5:
            utility = np.inf
        elif box_lost >= 5:
            utility = -np.inf

        return utility

    def count_chains(self, state: States) -> int:
        """
        Count the number of long chains in the given state.

        Args:
            state (States): Current game state.

        Returns:
            int: Number of long chains.
        """
        count_chains = 0
        chain_list = []

        for box_num in range(9):
            flag = False
            for chain in chain_list:
                if box_num in chain:
                    flag = True
                    break

            if not flag:
                chain_list.append([box_num])
                self.add_chain(state, chain_list, box_num)

        for chain in chain_list:
            if len(chain) >= 3:
                count_chains += 1

        return count_chains

    def add_chain(self, state: States, chain_list, box_num):
        """
        Add chains to the chain list in the given state.

        Args:
            state (States): Current game state.
            chain_list (List[List[int]]): List of chains.
            box_num (int): Box number.

        Returns:
            None
        """
        neighbors_num = [box_num - 1, box_num - 3, box_num + 1, box_num + 3]

        for idx in range(len(neighbors_num)):
            if (
                neighbors_num[idx] < 0
                or neighbors_num[idx] > 8
                or (idx % 2 == 0 and neighbors_num[idx] // 3 != box_num // 3)
            ):
                continue

            flag = False
            for chain in chain_list:
                if neighbors_num[idx] in chain:
                    flag = True
                    break

            if not flag and idx % 2 == 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.col_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain(state, chain_list, neighbors_num[idx])

            if not flag and idx % 2 != 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.row_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain(state, chain_list, neighbors_num[idx])

class DotsAndBoxes:
    def __init__(self, bot1, bot2):
        self.window = Tk()
        self.window.title("DotsAndBoxes")
        self.canvas = Canvas(
            self.window, width=size_of_board, height=size_of_board)
        self.canvas.pack()
        self.player1_starts = True
        self.refresh_board()

        self.bot1 = bot1
        self.bot2 = bot2
        self.play_again()

    def play_again(self):
        self.refresh_board()
        self.board_status = np.zeros(
            shape=(number_of_dots - 1, number_of_dots - 1))
        self.row_status = np.zeros(shape=(number_of_dots, number_of_dots - 1))
        self.col_status = np.zeros(shape=(number_of_dots - 1, number_of_dots))
        self.pointsScored = False

        # Input from user in form of clicks
        self.player1_starts = not self.player1_starts
        self.player1_turn = not self.player1_starts
        self.reset_board = False
        self.turntext_handle = []

        self.already_marked_boxes = []
        self.display_turn_text()

        self.turn()

    def mainloop(self):
        self.window.mainloop()

    def is_grid_occupied(self, logical_position, type):
        x = logical_position[0]
        y = logical_position[1]
        occupied = True

        if type == "row" and self.row_status[y][x] == 0:
            occupied = False
        if type == "col" and self.col_status[y][x] == 0:
            occupied = False

        return occupied

    def convert_grid_to_logical_position(self, grid_position):
        grid_position = np.array(grid_position)
        position = (grid_position - distance_between_dots / 4) // (
            distance_between_dots / 2
        )
        
        type = False
        logical_position = []
        if position[1] % 2 == 0 and (position[0] - 1) % 2 == 0:
            x = int((position[0] - 1) // 2)
            y = int(position[1] // 2)
            logical_position = [x, y]
            type = "row"
            # self.row_status[c][r]=1
        elif position[0] % 2 == 0 and (position[1] - 1) % 2 == 0:
            y = int((position[1] - 1) // 2)
            x = int(position[0] // 2)
            logical_position = [x, y]
            type = "col"

        return logical_position, type

    def pointScored(self):
        self.pointsScored = True

    def mark_box(self):
        boxes = np.argwhere(self.board_status == -4)
        for box in boxes:
            if list(box) not in self.already_marked_boxes and list(box) != []:
                self.already_marked_boxes.append(list(box))
                color = player1_color_light
                self.shade_box(box, color)

        boxes = np.argwhere(self.board_status == 4)
        for box in boxes:
            if list(box) not in self.already_marked_boxes and list(box) != []:
                self.already_marked_boxes.append(list(box))
                color = player2_color_light
                self.shade_box(box, color)

    def update_board(self, type, logical_position):
        x = logical_position[0]
        y = logical_position[1]
        val = 1
        playerModifier = 1
        if self.player1_turn:
            playerModifier = -1

        if y < (number_of_dots - 1) and x < (number_of_dots - 1):
            self.board_status[y][x] = (
                abs(self.board_status[y][x]) + val
            ) * playerModifier
            if abs(self.board_status[y][x]) == 4:
                self.pointScored()

        if type == "row":
            self.row_status[y][x] = 1
            if y >= 1:
                self.board_status[y - 1][x] = (
                    abs(self.board_status[y - 1][x]) + val
                ) * playerModifier
                if abs(self.board_status[y - 1][x]) == 4:
                    self.pointScored()

        elif type == "col":
            self.col_status[y][x] = 1
            if x >= 1:
                self.board_status[y][x - 1] = (
                    abs(self.board_status[y][x - 1]) + val
                ) * playerModifier
                if abs(self.board_status[y][x - 1]) == 4:
                    self.pointScored()

    def is_gameover(self):
        return (self.row_status == 1).all() and (self.col_status == 1).all()


    def make_edge(self, type, logical_position):
        if type == "row":
            start_x = (
                distance_between_dots / 2 +
                logical_position[0] * distance_between_dots
            )
            end_x = start_x + distance_between_dots
            start_y = (
                distance_between_dots / 2 +
                logical_position[1] * distance_between_dots
            )
            end_y = start_y
        elif type == "col":
            start_y = (
                distance_between_dots / 2 +
                logical_position[1] * distance_between_dots
            )
            end_y = start_y + distance_between_dots
            start_x = (
                distance_between_dots / 2 +
                logical_position[0] * distance_between_dots
            )
            end_x = start_x

        if self.player1_turn:
            color = player1_color
        else:
            color = player2_color
        self.canvas.create_line(
            start_x, start_y, end_x, end_y, fill=color, width=edge_width
        )

    def display_gameover(self):
        player1_score = len(np.argwhere(self.board_status == -4))
        player2_score = len(np.argwhere(self.board_status == 4))

        if player1_score > player2_score:
            # Player 1 wins
            text = "Winner: Player 1 "
            color = player1_color
        elif player2_score > player1_score:
            text = "Winner: Player 2 "
            color = player2_color
        else:
            text = "Its a tie"
            color = "gray"

        self.canvas.delete("all")
        self.canvas.create_text(
            size_of_board / 2,
            size_of_board / 3,
            font="cmr 60 bold",
            fill=color,
            text=text,
        )

        score_text = "Scores \n"
        self.canvas.create_text(
            size_of_board / 2,
            5 * size_of_board / 8,
            font="cmr 40 bold",
            fill=Green_color,
            text=score_text,
        )

        score_text = "Player 1 : " + str(player1_score) + "\n"
        score_text += "Player 2 : " + str(player2_score) + "\n"
        self.canvas.create_text(
            size_of_board / 2,
            3 * size_of_board / 4,
            font="cmr 30 bold",
            fill=Green_color,
            text=score_text,
        )
        self.reset_board = True

        score_text = "Click to play again \n"
        self.canvas.create_text(
            size_of_board / 2,
            15 * size_of_board / 16,
            font="cmr 20 bold",
            fill="gray",
            text=score_text,
        )

    def refresh_board(self):
        for i in range(number_of_dots):
            x = i * distance_between_dots + distance_between_dots / 2
            self.canvas.create_line(
                x,
                distance_between_dots / 2,
                x,
                size_of_board - distance_between_dots / 2,
                fill="gray",
                dash=(2, 2),
            )
            self.canvas.create_line(
                distance_between_dots / 2,
                x,
                size_of_board - distance_between_dots / 2,
                x,
                fill="gray",
                dash=(2, 2),
            )

        for i in range(number_of_dots):
            for j in range(number_of_dots):
                start_x = i * distance_between_dots + distance_between_dots / 2
                end_x = j * distance_between_dots + distance_between_dots / 2
                self.canvas.create_oval(
                    start_x - dot_width / 2,
                    end_x - dot_width / 2,
                    start_x + dot_width / 2,
                    end_x + dot_width / 2,
                    fill=dot_color,
                    outline=dot_color,
                )

    def display_turn_text(self):
        text = "Next turn: "
        if self.player1_turn:
            text += "Player1"
            color = player1_color
        else:
            text += "Player2"
            color = player2_color

        self.canvas.delete(self.turntext_handle)
        self.turntext_handle = self.canvas.create_text(
            size_of_board - 5 * len(text),
            size_of_board - distance_between_dots / 8,
            font="cmr 15 bold",
            text=text,
            fill=color,
        )

    def shade_box(self, box, color):
        start_x = (
            distance_between_dots / 2 + box[1] *
            distance_between_dots + edge_width / 2
        )
        start_y = (
            distance_between_dots / 2 + box[0] *
            distance_between_dots + edge_width / 2
        )
        end_x = start_x + distance_between_dots - edge_width
        end_y = start_y + distance_between_dots - edge_width
        self.canvas.create_rectangle(
            start_x, start_y, end_x, end_y, fill=color, outline=""
        )

    def display_turn_text(self):
        text = "Next turn: "
        if self.player1_turn:
            text += "Player1"
            color = player1_color
        else:
            text += "Player2"
            color = player2_color

        self.canvas.delete(self.turntext_handle)
        self.turntext_handle = self.canvas.create_text(
            size_of_board - 5 * len(text),
            size_of_board - distance_between_dots / 8,
            font="cmr 15 bold",
            text=text,
            fill=color,
        )

    def click(self, event):
        if not self.reset_board:
            grid_position = [event.x, event.y]
            logical_position, valid_input = self.convert_grid_to_logical_position(
                grid_position
            )
            self.update(valid_input, logical_position)
        else:
            self.canvas.delete("all")
            self.play_again()
            self.reset_board = False

    def update(self, valid_input, logical_position):
        if valid_input and not self.is_grid_occupied(logical_position, valid_input):
            self.window.unbind(LEFT_CLICK)
            self.update_board(valid_input, logical_position)
            self.make_edge(valid_input, logical_position)
            self.mark_box()
            self.refresh_board()
            self.player1_turn = (
                not self.player1_turn if not self.pointsScored else self.player1_turn
            )
            self.pointsScored = False

            if self.is_gameover():
                # self.canvas.delete("all")
                self.display_gameover()
                self.window.bind(LEFT_CLICK, self.click)
            else:
                self.display_turn_text()
                self.turn()

    def turn(self):
        current_bot = self.bot1 if self.player1_turn else self.bot2
        if current_bot is None:
            self.window.bind(LEFT_CLICK, self.click)
        else:
            self.window.after(BOT_TURN_INTERVAL_MS, self.agentTurn, current_bot)

    def agentTurn(self, bot):
        action = bot.best_decision(
            States(
                self.board_status.copy(),
                self.row_status.copy(),
                self.col_status.copy(),
                self.player1_turn,
            )
        )
        self.update(action.action_type, action.position)


def agent_A():
    
    bot = Bot()
    
    def choose_from(future_states):
        
        # future_states is list of States objects
        
        state = future_states[0] 
        actions = bot.create_required_move(state)
        
        # Evaluate states
        util_values = []
        for action in actions:
            new_state = bot.results(state, action)
            util_values.append(bot.calc_util_value(new_state))
        
        # Choose index of best state 
        best_index = np.argmax(np.array(util_values))
        
        return best_index
        
    return choose_from

def agent_A():
    bot = Bot()
    def choose_from(future_states):
        # future_states is list of States objects
        state = future_states[0] 
        actions = bot.create_required_move(state)
        
        # Evaluate states
        util_values = []
        for action in actions:
            new_state = bot.results(state, action)
            util_values.append(bot.calc_util_value(new_state))
        
        # Choose index of best state 
        best_index = np.argmax(np.array(util_values))
        
        return best_index
        
    return choose_from

def get_possible_states(state, bot):

    actions = bot.create_required_move(state)
    next_states = []

    for action in actions:
        next_state = bot.results(state, action)
        next_states.append(next_state)

    # Return list of possible next states
    return next_states, state.player1_turn 


def check_end_conditions(time_A, time_B, state):

    # Check if time exceeded for any player
    if time_A.elapsed_time() > TIMEOUT: 
        return True 
    if time_B.elapsed_time() > TIMEOUT:
        return True

    # Check if gameover based on current state  
    if is_gameover(state):
        return True

    # No ending condition met  
    return False

def is_gameover(state):
    return (state.row_status == 1).all() and (state.col_status == 1).all()

def get_future_states(current_state, bot):

    actions = bot.create_required_move(current_state)
    future_states = []

    for action in actions:
      
        next_state = bot.results(current_state, action)
        future_states.append(next_state)

    return future_states


def main():
    game = DotsAndBoxes( Bot(),None)
    game.mainloop()
    
if __name__ == "__main__":
    main()