class MaxList():
    def __init__(self, max_size):
        self.memory = []
        self.max_size = max_size

    def size(self):
        return len(self.memory)

    def append(self, value):
        if self.size() >= self.max_size:
            self.memory.pop(0)
        self.memory.append(value)
        return value

    def get(self):
        return self.memory