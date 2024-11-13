---
layout: post
category: Python编程
---

# Python类型检查
## 类型系统
> 在计算机编程中，类型系统是一个逻辑系统，包含一组规则，这些规则将称为类型的属性（例如整数、浮点、字符串）分配给每个“术语”（单词、短语或其他符号集）。类型系统的主要目的是减少由类型错误引发的程序错误。——wiki

所有编程语言都包含某种类型系统，该系统形式化了它可以使用哪些类别的对象以及如何处理这些类别。

类型系统主要分为两类：

* nominal system：基于类型名称的，它关注类型如何被命名，而不是它们的结构或成员。这意味着即使两个结构相似的类型（比如同样有相同的字段或方法）也可能不被认为是相同的类型，只要它们被分配了不同的名称。
* structural system：基于类型结构的，它关注的是类型的实际成员和结构。无论类型如何命名，只要它们具有相同的成员和结构，它们就被认为是相同的类型。这意味着即使类型被命名为不同的名称，只要它们的结构相似，编译器也可以将它们视为相同类型。

### 动态类型
python是一门动态类型语言，其有两个特点：

1. 解释器只在运行时进行类型检查；
2. 在生命周期中变量可以改变类型；

```python
>>> if False:
...     1 + "two"  # This line never runs, so no TypeError is raised
... else:
...     1 + 2
...
3
```
`if`的第一个分支不会运行，所以不会进行类型检查，也不会触发类型错误。

```Plain Text
>>> thing = "Hello"
>>> type(thing)
<class 'str'>

>>> thing = 28.1
>>> type(thing)
<class 'float'>
```
`thing`的类型允许变化。

### 静态类型
和动态类型相反，静态类型语言有两个相反特点：

1. 大部分静态类型语言，如C和JAVA，在编译阶段进行类型检查；
2. 尽管可能存在将变量转换为不同类型的机制，但通常不允许变量更改类型；

```java
String thing;
thing = "Hello";
thing = 28; // type error
```
第一行声明变量名`thing`在编译时绑定到 `String` 类型。该名称永远不能重新绑定为另一种类型。

在第二行中，为 `thing` 分配了一个值。永远不能为其分配非 `String` 对象的值。

### 鸭子类型
python是“鸭子类型“语言，这个绰号来源于一句话：“如果它走路像鸭子，嘎嘎叫像鸭子，那么它一定是鸭子。”

鸭子类型概念强调的是对象的行为/接口（method），而不是对象的血缘/类型（class）。

当对象具备了某些接口后，该对象就应当“视作“某个类型。

比如，python的内建函数`len()`可以应用于任何实现了`__len__` 方法的对象，而不是应用于某个类型的对象或其子类对象。

```python
>>> class TheHobbit:
...     def __len__(self):
...         return 95022
...
>>> the_hobbit = TheHobbit()
>>> len(the_hobbit)
95022
```
### 子类型(subtype)
形式上，如果以下两点条件成立，就说T是U的子类型：

1. T类型的每一个值（实例）都是U的值；
2. 支持U类型的所有函数（操作），都应当支持T类型。

例如，python中的bool是int的子类型，因为bool的两个值True和False，其实是0和1的别名，满足条件1；然后True和False支持int的所有操作，也满足条件2。

子类型的重要性在于子类型总是可以假装是其超类型。所以子类型可以通过超类型注解的检查：

```python
def double(number: int) -> int:
    return number * 2

print(double(True))  # Passing in bool instead of int
```
需要区分subtype和subclass。

> `bool` is a subtype of `int` because `bool` is a subclass of `int` .

> `int` is a subtype of `float`, but `int` is not a subclass of `float`.

在编程语言中，subtype 和 subclass 是两个相关但有所不同的概念。

subtype也称为接口继承，subclass称为实现继承或代码继承。

### 协变体、逆变体和不变体
这三个概念指的是如何从元素间的类型关系推导复合类型间的类型关系。

* Tuple是协变体，意味着其保留元素的类型层级关系，Tuple\[bool\] 是 Tuple\[int\] 的子类型，因为 bool 是 int 的子类型；
* List是不变体，意味着List\[bool\] 不是List\[int\]的subtype；
* Callable是逆变体，逆变意味着如果预期函数对 bool 进行操作，那么对 int 进行操作的函数也是可以接受的。

## 类型提示
类型提示，type hints，并不强制类型，仅仅只是建议。

关于注释的书写规范，依据PEP8:

* 冒号前没有空格，冒号后一个空格，如`text: str` ;
* 将参数注释与默认值组合时，请在 = 符号周围使用空格，如`align: bool = True` ;
* 箭头前后加空格，如：`def foo(...) -> str` ;

使用`mypy`可以进行静态类型检查。

### 代码注释 vs 类型提示
> “Code tells you how; Comments tell you why.”— *Jeff Atwood** (aka Coding Horror)*

类型提示相当于帮助代码自我注释。

二者都是为了代码的文档化。

注释为开发者，文档为使用者。

### 优缺点
类型提示优点：

* 帮助代码文档化
* 改善IDE或linter的性能
* 帮助维护和构建更清晰的代码架构

缺点：

* 写提示多花时间；
* 仅适用于高版本python；
* import typing增加消耗，其实微乎其微；

### 运行时使用注解
注解被存放在对象的`.__annotations__` 字典中，如有必要可以运行时访问。

`typing_extensions` 也提供了一些方法，可以在运行时进行类型检查。

### 存根文件
stub file专门用于注解代码，只给检查器使用，和\*.py 文件同名 \*.pyi。

## 注解方式
下文介绍不同python对象的添加类型提示的方式。

### 函数注解
```python
def func(arg: arg_type, optarg: arg_type = default) -> return_type:
    ...
```
运行时，可以通过函数的`__annotations__` 对象（字典）访问注解。

### 方法注解
和函数注解基本一致，区别在于self/cls不需要类型提示。

### \*args和\*\*kwargs注解
不确定参数的注解，不要当作集合变量来注解，而是当作一个参数来注解：

```python
class Game:
    def __init__(self, *names: str) -> None:
        """Set up the deck and deal cards to 4 players"""
        deck = Deck.create(shuffle=True)
        self.names = (list(names) + "P1 P2 P3 P4".split())[:4]
        self.hands = {
            n: Player(n, h) for n, h in zip(self.names, deck.deal(4))
        }
```
### 变量注解
```python
pi: float = 3.142

def circumference(radius: float) -> float:
    return 2 * pi * radius
```
其实静态类型检查器通过`pi = 3.142`也能推断出`pi`是`float`类型，此处不需要类型提示。

注解存放在module的`__annotations__` 对象中。

### Class作为注解类型
要将class用于类型提示，可直接使用class的名称。如：

```Plain Text
class Card:
    ...

class Deck:
    def __init__(self, cards: List[Card]) -> None:
        self.cards = cards
```
但是，当某个class未定义完成前，将其直接作为注解类型会出错(未来版本会支持)，比如：

```python
class Deck:
    @classmethod
    def create(cls, shuffle: bool = False) -> Deck:  # error：maybe work in Python 4.0
        """Create a new deck of 52 cards"""
        ...
```
此时应该使用单纯的文本字符串作为类型提示：

```python
class Deck:
    @classmethod
    def create(cls, shuffle: bool = False) -> "Deck":
        """Create a new deck of 52 cards"""
        cards = [Card(s, r) for r in Card.RANKS for s in Card.SUITS]
        if shuffle:
            random.shuffle(cards)
        return cls(cards)
```
### 类型注释
为了和python2兼容，类型提示可以通过类型注释的方式添加，而不是通过注解。

```python
import math

pi = 3.142  # type: float

def circumference(radius):
    # type: (float) -> float
    return 2 * math.pi * radius

def headline(text, width=80, fill_char="-"):
    # type: (str, int, str) -> str
    return f" {text.title()} ".center(width, fill_char)

# OR
def headline(
    text,           # type: str
    width=80,       # type: int
    fill_char="-",  # type: str
):                  # type: (...) -> str
    return f" {text.title()} ".center(width, fill_char)
```
区别在于，该注释仅供类型检查器使用，无法通过`__annotations__` 成员在运行时调用。

尽可能使用注解，而不是注释。

## Typing类型
下文介绍用于类型提示的类型对象，它们基本来自于`typing`模块，抽象类基本来自`collections.abc`模块，`typing`中有这些抽象类的别名封装。

### 序列和映射
内置的`list` 、`tuple`、`dict` 无法描述元素的类型。

```python
>>> from typing import Dict, List, Tuple, Sequence

>>> names: List[str] = ["Guido", "Jukka", "Ivan"]
>>> version: Tuple[int, int, int] = (3, 7, 1)
>>> options: Dict[str, bool] = {"centered": False, "capitalize": True}
>>> 
>>> def square(elems: Sequence[float]) -> List[float]:
...     return [x**2 for x in elems]
```
注意`Tuple`是不可变对象，元素数量固定，所以根据元素数量不同，对应不同元素数量的注解，如`Tuple[t_1, t_2, ..., t_n]`，表示n个元素的元素。

而`List` 是可变对象，元素数量可变，所以只有一种注解。如`List[T]` 。

相比之下，还有更泛化的类型`Sequence` ,正如鸭子类型关注行为，`Sequence` 注解表示该对象支持`len()`和`.__getitem__()` 。

这些注解类型可嵌套使用，如`List[Tuple[str, str]]` 。

### Callable
`Callable[[A1, A2, A3], Rt]` 表示三个参数类型A1...A3，一个返回值类型Rt。

`Callable[[], Rt]` 表示任意数量和类型的参数注解。

### 可选类型
对于多个可选类型的注解，使用`Union` 类型。

对于类型T和None二选一的注解，使用`Optional[T]`，等价于`Union[None, T]` 。

### Any类型
Python支持渐进式类型，可以逐步向Python代码添加类型提示。逐步类型化主要是由`Any`类型实现的。

类型检查器只会对不一致的类型发出警告。所以，你永远不会看到`Any`类型引起的类型错误。

这意味着你可以使用`Any`显式地回退到动态类型、描述在Python类型系统中难以描述的复杂类型或描述复合类型中的项。例如，一个带有字符串键并且可以取任何类型作为值的字典可以被标注为`Dict[str, Any]`。

请记住，如果你使用`Any`，静态类型检查器实际上不会进行任何类型检查。

```python
import random
from typing import Any, Sequence

def choose(items: Sequence[Any]) -> Any:
    return random.choice(items)
```
上述代码，实际上会丢失`choose` 函数的返回值类型检查。

### TypeVar
`TypeVar`变量是一种特殊变量，可以采用任何类型，具体取决于情景。

针对以上代码改进：

```python
# choose.py
import random
from typing import Sequence, TypeVar

Choosable = TypeVar("Choosable")
 
def choose(items: Sequence[Choosable]) -> Choosable:
    return random.choice(items)

names = ["Guido", "Jukka", "Ivan"]
reveal_type(names)

name = choose(names)
reveal_type(name)
```
不同于`Any` ，此时会进行类型检查，具体类型由检查器推导。

```Plain Text
mypy choose.py
choose.py:12: error: Revealed type is 'builtins.list[builtins.str*]'
choose.py:15: error: Revealed type is 'builtins.str*'
```
`TypeVar` 包装的类型本身可以进行限制，如：

```python
Choosable = TypeVar("Choosable", str, float)
```
将其限制为`str`和`float` 。

`TypeVar` 可以用来提示允许某个类型及其子类型，bound参数用来表明类型上界，如：

```python
TAnimal = TypeVar("TAnimal", bound="Animal")
class Animal:
    ...
    @classmethod
    def newborn(cls: Type[TAnimal], name: str) -> TAnimal:  # 如果直接注解为"Animal"，其子类调用可能提示缺少某属性
        return cls(name, date.today())


class Dog(Animal):
    def bark(self) -> None:
        print(f"{self.name} says woof!")

fido = Dog.newborn("Fido")
fido.bark()
```
如果`.newborn（）`返回值直接注解为` -> "Animal"`，类型推导fido引用的是`Animal`对象，此时`fido.bark()` 类型检查会报缺少bark方法的错误。

### 鸭子类型和协议
鸭子类型侧重的是行为，对行为的约束在Python中称作协议（Protocol），[PEP544](https://www.python.org/dev/peps/pep-0544/)引入。

协议对象明确了必须实现的方法，协议对象来自`collections.abc` 模块，`typing` 有别名封装。

`Container` ，抽象基类，提供`__contains__()`方法。

`Iterable` ，抽象基类，提供` __iter__()` 方法。

`Sized`，提供`__len__()` 方法。

`Callable` ，提供`__call__()`方法。

`Awaitable`, 

 `ContextManager` ,

用户可以通过继承`Protocol`来定义自己的协议类，用于类型检查。

### 类型别名
当注解类型嵌套太多，类型提示会变得不直观，可以通过定义类型别名来优化。

类型别名通过直接对象赋值来定义。

使用原始的类型嵌套：

```python
def deal_hands(
16    deck: List[Tuple[str, str]]
17) -> Tuple[
18    List[Tuple[str, str]],
19    List[Tuple[str, str]],
20    List[Tuple[str, str]],
21    List[Tuple[str, str]],
22]:
23    """Deal the cards in the deck into four hands"""
24    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])
```
使用类型别名

```python
Card = Tuple[str, str]
Deck = List[Card]

def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
    """Deal the cards in the deck into four hands"""
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])
```
### 无返回值的函数
没有返回值的函数（实际python会默认返回`None`），应当注解为返回None：

```python
def play(player_name: str) -> None:
    print(f"{player_name} plays")
    
ret_val = play("Filip")
```
这个注解有利于帮助类型检查器发现引用“无返回值“的错误。

#### None vs NoReturn
`typing.NoRetrun` 用于注解不期待返回值的函数，如总是引起异常的函数。

```python
from typing import NoReturn

def black_hole() -> NoReturn:
    raise Exception("There is no going back ...")
```
## 