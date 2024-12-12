class Heap(object):
    def __init__(self, heap_size):
        self.array = [0 for _ in range(heap_size)]
        self.size = heap_size
        self.cur_size = 0

    def __len__(self):
        return self.cur_size

    def max_size(self):
        return self.size

    def __repr__(self):
        return " ".join([str(x) for x in self.array[:self.cur_size]])

    def _heapify(self, idx):
        min_idx = idx
        left_idx = idx * 2 + 1
        right_idx = idx * 2 + 2
        if left_idx < self.cur_size and self.array[min_idx] > self.array[left_idx]:
            min_idx = left_idx
        if right_idx < self.cur_size and self.array[min_idx] > self.array[right_idx]:
            min_idx = right_idx
        if min_idx != idx:
            self.array[idx], self.array[min_idx] = self.array[min_idx], self.array[idx]
            self._heapify(min_idx)

    def insert(self, x):
        to_be_sub = -1
        if self.cur_size < self.size:
            self.array[self.cur_size] = x
            self.cur_size += 1
            index = int((self.cur_size - 1) / 2)
            while index >= 1:
                self._heapify(index)
                index = int((index - 1) / 2)
        else:
            if x > self.array[0]:
                to_be_sub = self.array[0]
                self.array[0] = x
                self._heapify(0)
        return to_be_sub
                
    def pop(self):
        if self.cur_size > 0:
            x = self.array[0]
            self.array[0] = self.array[self.cur_size - 1]
            self.cur_size -= 1
            self._heapify(0)
            return x
        raise ValueError("The heap is empty now")

    def is_insert(self, x):
        if self.cur_size < self.size or x > self.array[0]:
            return True
        return False



