# 8-puzzle problem 
# Kothamasu Jayachandra
#2110110293

import random
from collections import deque

class Puzzle:
    def __init__(self, state):
        self.state = list(state)

    def is_solvable(self):
        """Check if a puzzle is solvable.
        """
        inversions = 0
        for i in range(8):
            for j in range(i+1, 9):
                if self.state[j] and self.state[i] and self.state[i] > self.state[j]:
                    inversions += 1
        return inversions % 2 == 0
#function for playing 
    def play(self, k):
        for _ in range(k):
            move_direction = random.randint(0, 3)
            new_state = self.move(move_direction)
            if new_state is not None:
                self.state = list(new_state)
                
#function to move the puzzle
    def move(self, direction):
        index = self.state.index('0')
        if direction == 0:  
            if index % 3 != 0:
                self.state[index], self.state[index - 1] = self.state[index - 1], self.state[index]
            else:
                return None
        elif direction == 1:  
            if index % 3 != 2:
                self.state[index], self.state[index + 1] = self.state[index + 1], self.state[index]
            else:
                return None
        elif direction == 2:  
            if index - 3 >= 0:
                self.state[index], self.state[index - 3] = self.state[index - 3], self.state[index]
            else:
                return None
        elif direction == 3:  
            if index + 3 < len(self.state):
                self.state[index], self.state[index + 3] = self.state[index + 3], self.state[index]
            else:
                return None
        return ''.join(self.state)

# Function that itrates bfs 
def bfs(puzzle, goal):
    queue = deque([(puzzle, [])])
    direction_dict = {0: 'L', 1: 'R', 2: 'U', 3: 'D'}  
    visited = set()
    
    while queue:
        node, path = queue.popleft()
        visited.add(''.join(node.state))

        if ''.join(node.state) == goal:
            print("Shortest Path to goal (BFS):", ''.join(path))  
            return path

        for direction in range(4):
            new_puzzle = Puzzle(node.state[:])
            new_state = new_puzzle.move(direction)
            if new_state is not None and new_state not in visited:
                direction_str = 'LRUD'[direction]  
                queue.append((new_puzzle, path + [direction_str])) 
                
# function that itratres dls
def dls(puzzle, goal, limit):
    stack = [(puzzle, [], 0)]

    while stack:
        node, path, depth = stack.pop()

        if ''.join(node.state) == goal:
            print("Shortest Path to goal By (DLS):", ''.join(path))
            return path

        if depth < limit:
            for direction in range(4):
                new_puzzle = Puzzle(node.state[:])
                new_state = new_puzzle.move(direction)
                if new_state is not None:
                    direction_str = 'LRUD'[direction]
                    stack.append((new_puzzle, path + [direction_str], depth + 1))

    return None

#function that find ids
def ids(start, goal):
    limit = 0
    while True:
        path = dls(start, goal, limit)
        if path is not None:
            print("Shortest Path to goal By (IDS):", ''.join(map(str, path)))
            return path
        limit += 1
        
#function that find dfs for the puzzle
def dfs(puzzle, goal):
    stack = [(puzzle, [])]
    visited = set()

    while stack:
        node, path = stack.pop()
        visited.add(''.join(node.state))

        if ''.join(node.state) == goal:
            print("Shortest Path to goal By (DFS):", ''.join(path))
            return path

        for direction in range(4):
            new_puzzle = Puzzle(node.state[:])
            new_state = new_puzzle.move(direction)
            if new_state is not None and new_state not in visited:
                direction_str = 'LRUD'[direction]
                stack.append((new_puzzle, path + [direction_str]))

puzzle = Puzzle('123456780')
print("Initial state:", ''.join(puzzle.state))
puzzle.play(10)
print("State after playing:", ''.join(puzzle.state))
goal = '123456780'
dfs(puzzle, goal)
bfs(puzzle, goal)
ids(puzzle, goal)