import numpy as np
from game.game import Blockudoku

class BlockudokuEnv:
    def __init__(self):
        self.game = Blockudoku()
        self.action_space = 9 * 9 * 3  # 9x9 보드의 각 위치 * 3개 블록 선택
    
    def reset(self):
        self.game = Blockudoku()
        self.game.set_blocks(3)  # 3개의 블록 초기화
        return self._get_state()
    
    def step(self, action):
        block_index, row, col = divmod(action, 81)
        row, col = divmod(row, 9)
        
        reward = 0
        done = False
        
        if self.game.can_place_block(self.game.blocks[block_index], row, col):
            previous_score = self.game.score
            self.game.place_block(self.game.blocks[block_index], row, col)
            reward = self.game.score - previous_score
            
            # 사용한 블록 제거
            self.game.blocks[block_index] = None
            
            # 모든 블록을 사용했는지 확인
            if all(block is None for block in self.game.blocks):
                self.game.set_blocks(3)  # 새로운 3개의 블록 생성
            
            # 게임 오버 확인
            if self.game.is_game_over():
                done = True
                reward -= 50  # 게임 오버 페널티
        else:
            reward = -10  # 잘못된 위치에 블록을 놓으려고 시도한 경우 페널티
        
        return self._get_state(), reward, done, {"score": self.game.score}
    
    def _get_state(self):
        board_state = self.game.board.flatten()
        blocks_state = np.concatenate([self._pad_block(block) for block in self.game.blocks])
        return np.concatenate([board_state, blocks_state])
    
    def _pad_block(self, block):
        if block is None:
            return np.zeros(15)  # 5x3 크기로 변경
        padded = np.zeros((3, 5))  # 최대 3x5 크기로 변경
        block_rows, block_cols = block.shape
        padded[:block_rows, :block_cols] = block
        return padded.flatten()
    
    def render(self):
        print(self.game.board)
        print("Available blocks:")
        for i, block in enumerate(self.game.blocks):
            if block is not None:
                print(f"Block {i+1}:")
                print(block)
            else:
                print(f"Block {i+1}: Used")
        print(f"Score: {self.game.score}")
