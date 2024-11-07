import random
import numpy as np

from typing import List, Tuple


class Blockudoku:
    BLOCK_INDEX = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]
    COMBO_RATIO = 0.5
    
    def __init__(self) -> None:
        self.score: int = 0
        self.board: np.ndarray = np.zeros((9, 9), dtype=int)
        self.blocks: List[np.ndarray] = []
    
    def generate_block(self) -> np.ndarray:
        blocks = [
            # 1
            np.array([[1]]),
            
            # 2
            np.array([[1, 1]]),
            
            # 3
            np.array([[1, 1, 1]]),
            np.array([[1, 1], [1, 0]]),
            np.array([[1, 1], [0, 1]]),
            
            # 4
            np.array([[1, 1, 1, 1]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 1, 1], [1, 0, 0]]),
            np.array([[1, 1, 1], [0, 0, 1]]),
            np.array([[1, 1, 1], [0, 1, 0]]),
            np.array([[1, 1, 0], [0, 1, 1]]),
            np.array([[0, 1, 1], [1, 1, 0]]),
            
            # 5
            np.array([[1, 1, 1, 1, 1]]),
            np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
            np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]]),
            np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            np.array([[1, 0, 1], [1, 1, 1]]),
        ]
        
        return np.rot90(random.choices(blocks), random.randint(0, 3))
    
    def set_blocks(self, block_pool_length: int) -> None:
        self.blocks = [self.generate_block() for _ in range(block_pool_length)]
    
    def is_block_pool_empty(self) -> bool:
        return len(self.blocks) == 0
        
    def can_place_block(self, block: np.ndarray, row: int, col: int) -> bool:
        block_rows, block_cols = block.shape
        if row + block_rows > 9 or col + block_cols > 9: # out of bounds
            return False
        
        # check if the block can be placed in the board
        for i in range(block_rows):
            for j in range(block_cols):
                if block[i, j] == 1 and self.board[row + i, col + j] == 1:
                    return False
        return True
    
    def full_rows(self) -> List[int]:
        return [row for row in range(9) if np.all(self.board[row, :] == 1)]
    
    def full_cols(self) -> List[int]:
        return [col for col in range(9) if np.all(self.board[:, col] == 1)]
    
    def full_blocks(self) -> List[int]: # 1~9 in order grid
        return [self.BLOCK_INDEX.index((row // 3, col // 3)) for row in range(9) for col in range(9) if np.all(self.board[row:row+3, col:col+3] == 1)]
    
    def clear_row(self, row: int) -> None:
        self.board[row, :] = 0
        
    def clear_col(self, col: int) -> None:
        self.board[:, col] = 0
    
    def clear_block(self, block_index: int) -> None:
        row, col = self.BLOCK_INDEX[block_index]
        self.board[row:row+3, col:col+3] = 0

    def place_block(self, block: np.ndarray, row: int, col: int) -> None:
        
        if not self.can_place_block(block, row, col):
            raise ValueError("Block cannot be placed at the specified position")
        
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                if block[i, j] != 0: # 0 is empty -> later, it can be filled other number
                    self.board[row + i, col + j] = block[i, j]
        
        full_rows = self.full_rows()
        full_cols = self.full_cols()
        full_blocks = self.full_blocks()
        clear_count = 0
        
        for row in full_rows:
            self.clear_row(row)
            clear_count += 1
        for col in full_cols:
            self.clear_col(col)
            clear_count += 1
        for block in full_blocks:
            self.clear_block(block)
            clear_count += 1
            
        if clear_count > 1:
            self.score += int((clear_count) * 20 * self.COMBO_RATIO)
        elif clear_count == 1:
            self.score += 20
        else:
            self.score += block.sum()


    def is_game_over(self) -> bool:
        if self.is_block_pool_empty():
            return False
        
        for block in  self.blocks:
            for row in range(9):
                for col in range(9):
                    if self.can_place_block(block, row, col):
                        return False
        return True
    
    