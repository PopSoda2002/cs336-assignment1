from cs336_basics.utils import DoubleLinkedList

def test_double_linked_list():
    dll = DoubleLinkedList()
    dll.add_node(b"a")
    dll.add_node(b"b")
    dll.add_node(b"c")
    assert dll.head.next.value == b"a"
    assert dll.tail.prev.value == b"c"
    assert dll.head.next.next.value == b"b"
    assert dll.tail.prev.prev.value == b"b"

def test_double_linked_list_merge_pair():
    dll = DoubleLinkedList()
    dll.add_node(b"a")
    dll.add_node(b"b")
    dll.add_node(b"c")
    dll.merge_pair((b"a", b"b"))
    assert dll.head.next.value == b"ab"
    assert dll.tail.prev.value == b"c"
    assert dll.head.next.next.value == b"c"
    assert dll.tail.prev.prev.value == b"ab"
