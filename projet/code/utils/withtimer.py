import time


class Timer:
    depth = -1

    def __init__(self, name):
        self.name = name
        Timer.depth += 1

    def __enter__(self):
        self.start = time.clock()
        print('{}/ {}...'.format(' ' * 3 * Timer.depth, self.name))
        return self

    def __exit__(self, *args):
        print('{}\ {:.3f} s'.format(' ' * 3 * Timer.depth, time.clock() - self.start))
        Timer.depth -= 1
