import numpy as np

class CursorGrid:
    # Cursor uses global coordinates
    def __init__(self, buf_size, gx0, gy0, gx1, gy1, grid_x, grid_y, start_x, start_y, heat_rate, heat_range, move_rate, diffuse_rate):
        # Cursor data, gtl are global-to-local functions to convert cursor coords
        self.gtl_x = lambda x: grid_x * (x - gx0) / (gx1 - gx0)
        self.gtl_y = lambda y: grid_y * (y - gy0) / (gy1 - gy0)
        self.cursor_stream_x = [self.gtl_x(start_x)]
        self.cursor_stream_y = [self.gtl_y(start_y)]
        self.buf_size = buf_size
        self.size = 1

        # Smoothing and averaging variables for UI tricks
        self.total_x, self.total_y = self.gtl_x(start_x), self.gtl_x(start_x)
        self.smooth_x, self.smooth_y = self.gtl_x(start_x), self.gtl_x(start_x)
        self.move_rate = move_rate
        self.prev_x, self.prev_y = self.gtl_x(start_x), self.gtl_x(start_x)

        # The cursor affects values of a local heatmap array, which is mapped to global coords
        self.global_x0, self.global_y0 = gx0, gy0
        self.global_x1, self.global_y1 = gx1, gy1
        self.grid_x, self.grid_y = grid_x, grid_y
        self.block_x, self.block_y = (gx1 - gx0) / grid_x, (gy1 - gy0) / grid_y

        self.out = np.zeros((grid_x, grid_y))
        self.heat_range_x, self.heat_range_y = heat_range, heat_range
        self.heat_rate = heat_rate
        self.diffuse_rate = diffuse_rate


    # Increase the grid values near cx and cy, and globally cool down the map
    def heat(self, cx, cy):
        ax = int(max(0, (cx - self.heat_range_x)))
        bx = int(min(self.grid_x - 1, (cx + self.heat_range_x) + 1))
        ay = int(max(0, (cy - self.heat_range_y)))
        by = int(min(self.grid_y - 1, (cy + self.heat_range_y) + 1))
        for x in range(ax, bx + 1):
            for y in range(ay, by + 1):
                dx, dy = x - cx, y - cy
                r2 = dx * dx + dy * dy
                self.out[x, y] += self.heat_rate * np.random.random() / (r2 + 1)
        self.out *= (1 - self.diffuse_rate)


    # cursor_x and cursor_y enter as global coords
    def add_frame(self, cursor_x, cursor_y):
        cursor_x, cursor_y = self.gtl_x(cursor_x), self.gtl_y(cursor_y)
        # From now, cursor_x and cursor_y are in local coords
        self.cursor_stream_x.append(cursor_x)
        self.cursor_stream_y.append(cursor_y)
        self.total_x += cursor_x
        self.total_y += cursor_y
        # print(self.total)
        if self.size == self.buf_size:
            # print(self.stream[0] - frame)
            self.total_x -= self.cursor_stream_x.pop(0)
            self.total_y -= self.cursor_stream_y.pop(0)
        else:
            self.size += 1
        self.prev_x, self.prev_y = self.smooth_x, self.smooth_y
        self.smooth_x += (self.total_x / self.size - self.smooth_x) * self.move_rate
        self.smooth_y += (self.total_y / self.size - self.smooth_y) * self.move_rate

    # Return the local velocity
    def velocity(self):
        return self.smooth_x - self.prev_x, self.smooth_y - self.prev_y

    def global_velocity(self):
        return (self.smooth_x - self.prev_x) * self.block_x, (self.smooth_y - self.prev_y) * self.block_y