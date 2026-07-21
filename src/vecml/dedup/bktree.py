"""BK-tree over 64-bit hashes for sub-linear near-duplicate search.

A naive all-pairs Hamming comparison is O(n^2) and dies well before a million
files. A BK-tree indexes points in a metric space (Hamming here) so a
radius query visits only the branches whose stored distance overlaps the
search band, which keeps clustering feasible at corpus scale.
"""

from vecml.dedup.hashes import hamming


class BKTree:
    """Metric tree keyed on integer hashes under Hamming distance.

    Nodes are ``[key, payload, {edge_distance: child_node}]``. The tree is
    kept as plain lists (not a class per node) to stay light at millions of
    entries.
    """

    __slots__ = ("_root", "_size")

    def __init__(self):
        self._root = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, key: int, payload) -> None:
        self._size += 1
        if self._root is None:
            self._root = [key, payload, {}]
            return
        node = self._root
        while True:
            d = hamming(key, node[0])
            child = node[2].get(d)
            if child is None:
                node[2][d] = [key, payload, {}]
                return
            node = child

    def query(self, key: int, radius: int) -> list[tuple[object, int]]:
        """Return ``(payload, distance)`` for every entry within ``radius``."""
        if self._root is None:
            return []
        out: list[tuple[object, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = hamming(key, node[0])
            if d <= radius:
                out.append((node[1], d))
            lo, hi = d - radius, d + radius
            for edge, child in node[2].items():
                if lo <= edge <= hi:
                    stack.append(child)
        return out


class UnionFind:
    """Disjoint-set forest for single-linkage clustering of near-dupes."""

    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def clusters(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            groups.setdefault(self.find(i), []).append(i)
        return groups
