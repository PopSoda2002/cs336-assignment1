from typing import Optional

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

    def merge_pair(self, pair: tuple[bytes, bytes]) -> list[tuple[bytes, bytes]]:
        cur = self.head
        merged = False
        new_pairs = []
        while cur.next.val is not None:
            cur = cur.next
            if cur.val == pair[0] and cur.next.val == pair[1]:
                cur.val = pair[0] + pair[1]
                to_remove = cur.next
                cur.next = to_remove.next
                cur.next.prev = cur
                del to_remove
                self.size -= 1
                merged = True
                if cur.next.val is not None:
                    new_pairs.append((cur.val, cur.next.val))
                if cur.prev.val is not None:
                    new_pairs.append((cur.val, cur.prev.val))
        return new_pairs

    def __iter__(self):
        cur = self.head.next
        while cur.val is not None:
            yield cur.val
            cur = cur.next

    def __str__(self) -> str:
        return "[" + ", ".join(repr(b) for b in self) + "]"

    def get_pair(self) -> tuple[bytes, bytes]:
        cur = self.head
        while cur.next.val is not None:
            cur = cur.next
            if cur.val is not None and cur.next.val is not None:
                return (cur.val, cur.next.val)
        return None
