from typing import Optional
from collections import defaultdict

class Node:
    def __init__(self, val: Optional[bytes]):
        self.val: Optional[bytes] = val
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None

class DoublyLinkedList:
    def __init__(self, value: Optional[bytes] = None, freq: int = 0):
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
        self.freq = freq

        if value is not None:
            for i in range(len(value)):
                self.add_node(bytes([value[i]]))
    
    def add_node(self, value: bytes) -> None:
        new_node = Node(value)
        front_node = self.tail.prev
        front_node.next = new_node
        new_node.prev = front_node
        new_node.next = self.tail
        self.tail.prev = new_node
        self.size += 1
        return

    def merge_pair_and_get_deltas(self, pair: tuple[bytes, bytes]) -> dict[tuple[bytes, bytes], int]:
        """
        把所有 (A,B) 合并为 AB 返回相邻对频次的差分
        -1: (prev,A)、(A,B)、(B,next) +1: (prev,AB)、(AB,next)
        """
        A, B = pair
        deltas = defaultdict(int)
        cur = self.head.next

        while cur is not self.tail and cur.next is not self.tail:
            if cur.val == A and cur.next.val == B:
                left = cur.prev.val if cur.prev is not self.head else None
                right = cur.next.next.val if cur.next.next is not self.tail else None

                # 旧对减少
                if left is not None:
                    deltas[(left, A)] -= 1
                deltas[(A, B)] -= 1
                if right is not None:
                    deltas[(B, right)] -= 1

                # 合并为 AB
                cur.val = A + B
                to_remove = cur.next
                cur.next = to_remove.next
                cur.next.prev = cur
                self.size -= 1
                # del to_remove   # 非必须

                # 新对增加
                if left is not None:
                    deltas[(left, cur.val)] += 1
                if right is not None:
                    deltas[(cur.val, right)] += 1

                # 合并后，cur 继续向右扫描，不回退
            else:
                cur = cur.next

        return deltas

    def __iter__(self):
        cur = self.head.next
        while cur.val is not None:
            yield cur.val
            cur = cur.next

    def __str__(self) -> str:
        return "[" + ", ".join(repr(b) for b in self) + "]"

    def get_pairs(self) -> list[tuple[bytes, bytes]]:
        cur = self.head
        res = []
        while cur.next.val is not None:
            cur = cur.next
            if cur.val is not None and cur.next.val is not None:
                res.append((cur.val, cur.next.val))
        return res
