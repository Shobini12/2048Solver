# Shobini Iyer (si2449)

import time
import math
from BaseAI import BaseAI

class IntelligentAgent(BaseAI):
    def __init__(self):
        self.time_limit = 0.18  
        self.start_time = 0.0

        # Heuristic weights 
        self.empty_weight = 135000.0
        self.smoothness_weight = 3.0
        self.monotonicity_weight = 1200.0
        self.max_tile_weight = 1.0
        self.position_weight = 100.0

        #give corner positions more weight
        self.position_matrix = [
            [32768, 16384, 8192, 4096],
            [256,   512,   1024, 2048],
            [128,   64,    32,   16],
            [1,     2,     4,    8]
        ]

    def getMove(self, grid):
        self.start_time = time.monotonic()
        moves = grid.getAvailableMoves()
        if not moves:
            return None
        best_move = moves[0][0]
        best_score = float('-inf')
        for move, child in moves:
            if time.monotonic() - self.start_time > self.time_limit:
                break
            score = self.expectiminimax(child, 4, True, float('-inf'), float('inf'))
            if score is not None and score > best_score:
                best_score = score
                best_move = move
        return best_move

   # Expeciminimax
    def expectiminimax(self, grid, depth, is_chance, alpha, beta):
        if time.monotonic() - self.start_time > self.time_limit:
            return None
        if depth == 0 or not grid.canMove():
            return self.evaluate(grid)
        if is_chance:
            return self.chance_node(grid, depth, alpha, beta)
        else:
            return self.max_node(grid, depth, alpha, beta)

    def max_node(self, grid, depth, alpha, beta):
        max_score = float('-inf')
        moves = grid.getAvailableMoves()
        if not moves:
            return self.evaluate(grid)

        moves.sort(key=lambda m: -self.quick_eval(m[1]))
        for _, child in moves:
            if time.monotonic() - self.start_time > self.time_limit:
                return None
            score = self.expectiminimax(child, depth - 1, True, alpha, beta)
            if score is None:
                return None
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            if beta <= alpha: #alpha beta pruning
                break
        return max_score

    def chance_node(self, grid, depth, alpha, beta):
        empty = grid.getAvailableCells()
        if not empty:
            return self.evaluate(grid)
        n = len(empty)
        expected = 0.0

        if n > 6:
            empty = self.sample_cells(grid, empty, 6)
            n = len(empty)
        for cell in empty:
            if time.monotonic() - self.start_time > self.time_limit:
                return None
            g2 = grid.clone(); g2.insertTile(cell, 2)
            s2 = self.expectiminimax(g2, depth - 1, False, alpha, beta)
            if s2 is None:
                return None
            g4 = grid.clone(); g4.insertTile(cell, 4)
            s4 = self.expectiminimax(g4, depth - 1, False, alpha, beta)
            if s4 is None:
                return None
            expected += (0.9 * s2 + 0.1 * s4) / n
        return expected

    def sample_cells(self, grid, cells, k):
        max_tile = grid.getMaxTile()
        max_pos = None
        for i in range(grid.size):
            for j in range(grid.size):
                if grid.map[i][j] == max_tile:
                    max_pos = (i, j)
                    break
            if max_pos:
                break
        def priority(cell):
            i, j = cell
            corner = min(i, grid.size - 1 - i) + min(j, grid.size - 1 - j)
            max_dist = abs(i - max_pos[0]) + abs(j - max_pos[1]) if max_pos else 0
            return (corner, max_dist)
        cells.sort(key=priority)
        return cells[:k]

    def quick_eval(self, grid):
        empty = len(grid.getAvailableCells())
        max_tile = grid.getMaxTile()
        max_corner = 1 if self.is_max_in_corner(grid, max_tile) else 0
        return empty * 1000 + max_corner * 10000 + max_tile

    def evaluate(self, grid):
        #SOURCE: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
        empty = len(grid.getAvailableCells())
        smooth = self.smoothness(grid)
        mono = self.monotonicity(grid)
        max_tile = grid.getMaxTile()
        pos_score = self.position_score(grid)
        score = (
            self.empty_weight * empty
            + self.smoothness_weight * smooth
            + self.monotonicity_weight * mono
            + self.max_tile_weight * max_tile
            + self.position_weight * pos_score)
        return score

    def smoothness(self, grid):
        smooth = 0.0
        for i in range(grid.size):
            for j in range(grid.size):
                v = grid.map[i][j]
                if v:
                    lv = math.log2(v)
                    if j + 1 < grid.size and grid.map[i][j + 1]:
                        smooth -= abs(lv - math.log2(grid.map[i][j + 1]))
                    if i + 1 < grid.size and grid.map[i + 1][j]:
                        smooth -= abs(lv - math.log2(grid.map[i + 1][j]))
        return smooth

    def monotonicity(self, grid):
        totals = [0, 0, 0, 0]
        for i in range(grid.size):
            current = 0
            next_idx = 1
            while next_idx < grid.size:
                while next_idx < grid.size and grid.map[i][next_idx] == 0:
                    next_idx += 1
                if next_idx >= grid.size:
                    break
                cur = grid.map[i][current]
                nxt = grid.map[i][next_idx]
                if cur:
                    if cur > nxt:
                        totals[0] += math.log2(cur) - math.log2(nxt) if nxt else math.log2(cur)
                    elif nxt > cur:
                        totals[1] += math.log2(nxt) - math.log2(cur)
                current = next_idx
                next_idx += 1
        for j in range(grid.size):
            current = 0
            next_idx = 1
            while next_idx < grid.size:
                while next_idx < grid.size and grid.map[next_idx][j] == 0:
                    next_idx += 1
                if next_idx >= grid.size:
                    break
                cur = grid.map[current][j]
                nxt = grid.map[next_idx][j]
                if cur:
                    if cur > nxt:
                        totals[2] += math.log2(cur) - math.log2(nxt) if nxt else math.log2(cur)
                    elif nxt > cur:
                        totals[3] += math.log2(nxt) - math.log2(cur)
                current = next_idx
                next_idx += 1
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def position_score(self, grid):
        score = 0.0
        for i in range(grid.size):
            for j in range(grid.size):
                v = grid.map[i][j]
                if v:
                    score += v * self.position_matrix[i][j]
        return score

    def is_max_in_corner(self, grid, max_tile):
        return (
            grid.map[0][0] == max_tile or
            grid.map[0][grid.size - 1] == max_tile or
            grid.map[grid.size - 1][0] == max_tile or
            grid.map[grid.size - 1][grid.size - 1] == max_tile
        )