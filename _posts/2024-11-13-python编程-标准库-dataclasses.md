---
layout: post
category: Python编程
---

# 标准库: dataclasses
该模块提供了一个装饰器和函数，可以根据用户定义的类，自动生成`__init__` 、`__repr__`等方法。

`dataclasses`模块中`dataclass`函数的实现：

```python
def dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False, match_args=True,
              kw_only=False, slots=False):
    """Returns the same class as was passed in, with dunder methods
    added based on the fields defined in the class.
    Examines PEP 526 __annotations__ to determine fields.
    If init is true, an __init__() method is added to the class. If
    repr is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method function is added. If frozen is true, fields may
    not be assigned to after instance creation. If match_args is true,
    the __match_args__ tuple is added. If kw_only is true, then by
    default all fields are keyword-only. If slots is true, an
    __slots__ attribute is added.
    """
    (...)
```
PEP 526规定的注释方式，被注释的简单名称会放到`__annotations__` 映射中。

利用被注释的成员变量（`field`）去生成变量和对应方法，减少劳动。

这也意味着没有被类型注释的成员变量不会被考虑。

```python
from dataclasses import dataclass

@dataclass
class InventoryItem:
    '''Class for keeping track of an item in inventory.'''
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
```
相当于生成以下方法：

```python
def __init__(self, name: str, unit_price: float, quantity_on_hand: int = 0) -> None:
    self.name = name
    self.unit_price = unit_price
    self.quantity_on_hand = quantity_on_hand
def __repr__(self):
    return f'InventoryItem(name={self.name!r}, unit_price={self.unit_price!r}, quantity_on_hand={self.quantity_on_hand!r})'
def __eq__(self, other):
    if other.__class__ is self.__class__:
        return (self.name, self.unit_price, self.quantity_on_hand) == (other.name, other.unit_price, other.quantity_on_hand)
    return NotImplemented
def __ne__(self, other):
    if other.__class__ is self.__class__:
        return (self.name, self.unit_price, self.quantity_on_hand) != (other.name, other.unit_price, other.quantity_on_hand)
    return NotImplemented
def __lt__(self, other):
    if other.__class__ is self.__class__:
        return (self.name, self.unit_price, self.quantity_on_hand) < (other.name, other.unit_price, other.quantity_on_hand)
    return NotImplemented
def __le__(self, other):
    if other.__class__ is self.__class__:
        return (self.name, self.unit_price, self.quantity_on_hand) <= (other.name, other.unit_price, other.quantity_on_hand)
    return NotImplemented
def __gt__(self, other):
    if other.__class__ is self.__class__:
        return (self.name, self.unit_price, self.quantity_on_hand) > (other.name, other.unit_price, other.quantity_on_hand)
    return NotImplemented
def __ge__(self, other):
    if other.__class__ is self.__class__:
        return (self.name, self.unit_price, self.quantity_on_hand) >= (other.name, other.unit_price, other.quantity_on_hand)
    return NotImplemented
```
包含了构造方法、repr方法、比较方法、hash方法等；

## field函数
`dataclasses.field` 方法提供接口，允许用户定制每个field的创建和相关设置，比如设置默认值，不将某个field作为实例变量等。

重要的参数：

* default：默认值；
* defualt\_factory: 默认工厂函数，必须是0参数的函数，用于每次创建默认值；
* init：指定该field是否被构造函数初始化；
* repr：指定该field是否包含在repr方法里；
* compare：指定该filed是否用于顺序比较；

注意存在`dataclasses.Field` 对象，该对象不会直接实例化，而通过`fields()` 函数返回。

```python
@dataclass
class C:
    x: int
    y: int = field(repr=False)
    z: int = field(repr=False, default=10)
    t: int = 20
```
## 可变对象的默认值
因为引用的关系，当类成员的默认值是不可变对象时，所有实例的该类成员指向同一个不可变对象。

```python
class C:
    x = []
    def add(self, element):
        self.x.append(element)

o1 = C()
o2 = C()
o1.add(1)
o2.add(2)
assert o1.x == [1, 2] # Pass
assert o1.x is o2.x # Pass
```
当使用`dataclass` 装饰器生成类时，同样会面临该问题。

```python
@dataclass
class D:
    x: List = []
    def add(self, element):
        self.x += element
```
相当于

```python
class D:
    x = []
    def __init__(self, x=x):  # def __init__(self, x=[]):
        self.x = x
    def add(self, element):
        self.x += element

assert D().x is D().x  # Pass
```
同时这也是提醒我们，方法的默认参数不应该是可变参数。

为了避免该情况，使用`field` 的默认方法创建默认值。

```python
@dataclass
class D:
    x: list = field(default_factory=list)

assert D().x is not D().x
```
