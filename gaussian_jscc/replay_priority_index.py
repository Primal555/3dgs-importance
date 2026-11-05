"""Deterministic order-statistic tree for mutable replay priorities."""


def _rank(index):
    mask = (1 << 64)-1
    value = (index+0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30))*0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27))*0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


class _Node:
    __slots__ = ('key', 'rank', 'left', 'right', 'count', 'mass')

    def __init__(self, key):
        self.key, self.rank = key, _rank(key[1])
        self.left = self.right = None
        self.count, self.mass = 1, key[0]


def _count(node):
    return node.count if node else 0


def _mass(node):
    return node.mass if node else 0.


def _refresh(node):
    node.count = 1+_count(node.left)+_count(node.right)
    node.mass = _mass(node.left)+node.key[0]+_mass(node.right)
    return node


def _split(node, key):
    if node is None:
        return None, None
    if node.key < key:
        node.right, right = _split(node.right, key)
        return _refresh(node), right
    left, node.left = _split(node.left, key)
    return left, _refresh(node)


def _insert(node, new):
    if node is None:
        return new
    if new.rank < node.rank:
        new.left, new.right = _split(node, new.key)
        return _refresh(new)
    if new.key < node.key:
        node.left = _insert(node.left, new)
    else:
        node.right = _insert(node.right, new)
    return _refresh(node)


def _merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left.rank < right.rank:
        left.right = _merge(left.right, right)
        return _refresh(left)
    right.left = _merge(left, right.left)
    return _refresh(right)


def _erase(node, key):
    if node is None:
        raise KeyError(key)
    if node.key == key:
        return _merge(node.left, node.right)
    if key < node.key:
        node.left = _erase(node.left, key)
    else:
        node.right = _erase(node.right, key)
    return _refresh(node)


class PriorityIndex:
    def __init__(self):
        self.root = None

    @property
    def total(self):
        return _mass(self.root)

    def add(self, index, value):
        self.root = _insert(self.root, _Node((float(value), index)))

    def update(self, index, old, new):
        self.root = _erase(self.root, (float(old), index))
        self.add(index, new)

    def median(self):
        if self.root is None:
            return 1.
        rank = (self.root.count-1)//2
        node = self.root
        while node:
            count = _count(node.left)
            if rank == count:
                return node.key[0]
            if rank < count:
                node = node.left
            else:
                rank -= count+1
                node = node.right
        raise RuntimeError('invalid priority rank')

    def weighted(self, unit):
        if self.total <= 0:
            raise ValueError('positive priority mass required')
        mass = min(unit, 1.-2**-53)*self.total
        node = self.root
        while node:
            left = _mass(node.left)
            if mass < left:
                node = node.left
            elif mass < left+node.key[0]:
                return node.key[1]
            else:
                mass -= left+node.key[0]
                node = node.right
        raise RuntimeError('invalid priority mass')
