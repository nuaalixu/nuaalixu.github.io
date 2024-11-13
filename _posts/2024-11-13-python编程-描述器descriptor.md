---
layout: post
category: Python编程
---

# 描述器 descriptor
## 定义
任何定义了 `__get__()`, `__set__()` 或 `__delete__()` 方法的对象。当一个**类属性**为描述器时，它的特殊绑定行为就会在属性查找时被触发。

注意：类属性为描述器时，特殊行为才会触发。

描述器是一个强大而通用的协议。 它们是属性、方法、静态方法、类方法和 `super()` 背后的实现机制。 它们在 Python 内部被广泛使用。 描述器简化了底层的 C 代码并为 Python 的日常程序提供了一组灵活的新工具。

## 描述器协议
`descr.__get__(self, obj, type=None) -> value`

`descr.__set__(self, obj, value) -> None`

`descr.__delete__(self, obj) -> None`

描述器的方法就这些。一个对象只要定义了以上方法中的任何一个，就被视为描述器，并在被作为属性时覆盖其默认行为。

如果一个对象定义了 `__set__()` 或 `__delete__()`，则它会被视为数据描述器。 仅定义了 `__get__()` 的描述器称为非数据描述器（它们经常被用于方法，但也可以有其他用途）。

数据和非数据描述器的不同之处在于，如何计算实例字典中条目的替代值。如果实例的字典具有与数据描述器同名的条目，则数据描述器优先。如果实例的字典具有与非数据描述器同名的条目，则该字典条目优先。

为了使数据描述器成为只读的，应该同时定义 `__get__()` 和 `__set__()` ，并在 `__set__()` 中引发 `AttributeError` 。用引发异常的占位符定义 `__set__()` 方法使其成为数据描述器。

## 描述器的调用概述
描述器可以通过 `descr.__get__(obj)` 或 `descr.__get__(None, cls)` 直接调用。但更常见的是通过属性访自动调用描述器。

表达式 `obj.x` 在命名空间的链中查找`obj`的属性 `x`。如果搜索在实例 `__dict__` 之外找到描述器，则根据下面列出的优先级规则调用其 `__get__()` 方法。

调用的细节取决于 `obj` 是对象、类还是超类的实例。

### **通过实例调用**
实例查找通过命名空间链进行扫描，数据描述器的优先级最高，其次是实例变量、非数据描述器、类变量，最后是 `__getattr__()` （如果存在的话）。

如果 `a.x` 找到了一个描述器，那么将通过 `desc.__get__(a, type(a))` （也可以不传入类）调用它。

点运算符的查找逻辑在 `object.__getattribute__()` 中。这里是一个等价的纯 Python 实现：

```python
def find_name_in_mro(cls, name, default):
    "Emulate _PyType_Lookup() in Objects/typeobject.c"
    for base in cls.__mro__:
        if name in vars(base):
            return vars(base)[name]
    return default

def object_getattribute(obj, name):
    "Emulate PyObject_GenericGetAttr() in Objects/object.c"
    null = object()
    objtype = type(obj)
    cls_var = find_name_in_mro(objtype, name, null)  # 从objtype可以看到描述器要在类中才生效
    descr_get = getattr(type(cls_var), '__get__', null)
    if descr_get is not null:
        if (hasattr(type(cls_var), '__set__')
            or hasattr(type(cls_var), '__delete__')):
            return descr_get(cls_var, obj, objtype)     # data descriptor
    if hasattr(obj, '__dict__') and name in vars(obj):
        return vars(obj)[name]                          # instance variable
    if descr_get is not null:
        return descr_get(cls_var, obj, objtype)         # non-data descriptor
    if cls_var is not null:
        return cls_var                                  # class variable
    raise AttributeError(name)
```
### **通过类调用**
像 `A.x` 这样的点操作符查找的逻辑在 `type.__getattribute__()` 中。步骤与 `object.__getattribute__()` 相似，但是实例字典查找改为搜索类的 [method resolution order](https://docs.python.org/zh-cn/3/glossary.html#term-method-resolution-order)。

如果找到了一个描述器，那么将通过 `desc.__get__(None, A)` 调用它。

**通过****super**** 调用**

super 的点操作符查找的逻辑在 `super()` 返回的对象的 `__getattribute__()` 方法中。

类似 `super(A, obj).m` 形式的点分查找将在 `obj.__class__.__mro__` 中搜索紧接在 `A` 之后的基类 `B`，然后返回 `B.__dict__['m'].__get__(obj, A)`。如果 `m` 不是描述器，则直接返回其值。

## 总结
descriptor 就是任何一个定义了` __get__()`，`__set__()` 或 `__delete__()`的对象。可选地，描述器可以具有 `__set_name__()` 方法。这仅在描述器需要知道创建它的类或分配给它的类变量名称时使用。（即使该类不是描述器，只要此方法存在就会调用。）

在属性查找期间，描述器由点运算符调用。如果使用 `vars(some_class)[descriptor_name]`间接访问描述器，则返回描述器实例而不调用它。

描述器仅在用作**类变量**时起作用。放入实例时，它们将失效。

描述起的`__get__` 方法可以被实例和类调用触发，但`__set__` 方法必须被实例调用才触发。

描述器的主要目的是提供一个挂钩，允许存储在类变量中的对象控制在属性查找期间发生的情况。

传统上，调用类控制查找过程中发生的事情。描述器反转了这种关系，并允许正在被查询的数据对此进行干涉。

描述器的使用贯穿了整个语言。**就是它让函数变成绑定方法**。常见工具诸如 `classmethod()`， `staticmethod()`，`property()` 和 `functools.cached_property()` 都作为描述器实现。

## 示例
### **实例方法绑定**
实例方法绑定指的是将python函数（function对象）和某一个类（class）的实例进行绑定。

function是一个描述器，利用描述器协议，将第一个参数和一个对象实例绑定。

```python
class function:
    ...

    def __get__(self, obj, objtype=None):
        "Simulate func_descr_get() in Objects/funcobject.c"
        if obj is None:
            return self
        return MethodType(self, obj)
```
`MethodType`是method的类，作用是将传入func的第一个参数和传入的实例绑定，其python等价实现：

```python
class MethodType:
    "Emulate PyMethod_Type in Objects/classobject.c"

    def __init__(self, func, obj):
        self.__func__ = func
        self.__self__ = obj

    def __call__(self, *args, **kwargs):
        func = self.__func__
        obj = self.__self__
        return func(obj, *args, **kwargs)
```
`MethodType`含有`__get__` 方法，但只返还自身。意味着method无法二次绑定。

通过实例`obj.func(*)`调用时，借助描述器协议，最终返还的是`func(obj, *args, **kwargs)`的结果，即实现了func和obj的绑定。

描述器协议对绑定method的演示：

```python
class D
    def f(self, x):
        return x

D.__dict__['f']  #不会触发Function的__get__方法
# <function D.f at 0x00C45070>

D.f  # 通过类调用，触发Function的__get__方法，但返回self
# <function D.f at 0x00C45070>

d = D()
d.f. # 通过实例调用，触发Function的__get__方法，返回绑定的MethodType对象
# <bound method D.f of <__main__.D object at 0x00B18C90>>

d.f.__func__
# <function D.f at 0x00C45070>
d.f.__self__
# <__main__.D object at 0x1012e1f98>
```
### **静态方法**
纯Python版本的`staticmethod`定义如下：

```python
class StaticMethod:
    "Emulate PyStaticMethod_Type() in Objects/funcobject.c"

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, objtype=None):
        return self.f

    def __call__(self, *args, **kwds):
        return self.f(*args, **kwds)
```
无论是实例调用还是类调用，静态函数都可以进行相同的访问。

### **类方法绑定**
`classmethod()` 如果用纯python的代码表示，应该如下：

```python
class ClassMethod(object):
"Emulate PyClassMethod_Type() in Objects/funcobject.c"

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        def newfunc(*args):
            return self.f(klass, *args)
        return newfunc
```
当通过dot运算符调用某个类方法时，会调用`ClassMethod` 实例的`__get__`方法，根据描述器的调用方式可知，不论是通过实例调用，还是通过类调用，`ClassMethod` 实例的`__get__`方法都会将类作为函数f的第一个实参。

### **托管属性**
描述器的一种流行用法是托管对实例数据的访问。描述器被分配给类字典中的公开属性，而实际数据作为私有属性存储在实例字典中。当访问公开属性时，会触发描述器的 `__get__()` 和 `__set__()` 方法。

在下面的例子中，`age` 是公开属性，`_age` 是私有属性。当访问公开属性时，描述器会记录下查找或更新的日志：

```python
import logging

logging.basicConfig(level=logging.INFO)

class LoggedAgeAccess:

    def __get__(self, obj, objtype=None):
        value = obj._age
        logging.info('Accessing %r giving %r', 'age', value)
        return value

    def __set__(self, obj, value):
        logging.info('Updating %r to %r', 'age', value)
        obj._age = value  # 手动添加了_age为实例属性

class Person:

    age = LoggedAgeAccess()             # Descriptor instance

    def __init__(self, name, age):
        self.name = name                # Regular instance attribute
        self.age = age                  # Calls __set__()

    def birthday(self):
        self.age += 1                   # Calls both __get__() and __set__()
```
此示例的一个主要问题是私有名称 `_age`在类 `LoggedAgeAccess` 中是硬耦合的。这意味着每个实例只能有一个用于记录的属性，并且其名称不可更改。

### **定制名称**
当一个类使用描述器时，它可以告知每个描述器使用了什么变量名。

在此示例中， `Person` 类具有两个描述器实例`name` 和 `age`。当类 `Person` 被定义的时候，他回调了 `LoggedAccess` 中的 `__set_name__()` 来记录字段名称，让每个描述器拥有自己的 `public_name`和`private_name`：

```python
import logging

logging.basicConfig(level=logging.INFO)

class LoggedAccess:

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        logging.info('Accessing %r giving %r', self.public_name, value)
        return value

    def __set__(self, obj, value):
        logging.info('Updating %r to %r', self.public_name, value)
        setattr(obj, self.private_name, value)

class Person:

    name = LoggedAccess()                # First descriptor instance
    age = LoggedAccess()                 # Second descriptor instance

    def __init__(self, name, age):
        self.name = name                 # Calls the first descriptor
        self.age = age                   # Calls the second descriptor

    def birthday(self):
        self.age += 1
```
这样同一个描述符类，可以有多个实例分别托管不同的私有属性。

### property装饰属性
许多面向对象编程语言中都使用 Getter（也称为“访问器”）和 Setter（又名“修改器”）来确保数据封装的原则。`@property` 的用法，提供了一种pythonic的方式来达到Getter和Setter的效果。

`property` 是一个描述器，通过装饰器的方式，将实例方法装饰为属性，进行数据封装。

假设类“Ourclass“有一个成员变量“OurAtt”，在赋值时需要对所传递的值进行范围限定。

传统的实现方式，是将“OurAtt”设置为私有成员变量，然后设置成员方法如“setOurAtt”，对变量进行赋值。利用`property` 作为装饰器，实现更加简单。

```python
class OurClass:

    def __init__(self, a):
        self.OurAtt = a

    @property
    def OurAtt(self):
        return self.__OurAtt

    @OurAtt.setter
    def OurAtt(self, val):
        if val < 0:
            self.__OurAtt = 0
        elif val > 1000:
            self.__OurAtt = 1000
        else:
            self.__OurAtt = val


x = OurClass(10)
print(x.OurAtt)
# 10
```
不要盲目的使用`@property` 替代Getter和Setter风格，当：1.动态计算或验证更复杂；2.外部API兼容；3.赋值需要额外的参数；等情况时，请使用传统的风格。

### **动态查找**
有趣的描述器通常运行计算而不是返回常量：

```python
import os

class DirectorySize:

    def __get__(self, obj, objtype=None):
        return len(os.listdir(obj.dirname))

class Directory:

    size = DirectorySize()              # Descriptor instance

    def __init__(self, dirname):
        self.dirname = dirname          # Regular instance attribute
```
交互式会话显示查找是动态的，每次都会计算不同的，经过更新的返回值。

```python
>>> s = Directory('songs')
>>> g = Directory('games')
>>> s.size                              # The songs directory has twenty files
20
>>> g.size                              # The games directory has three files
3
>>> os.remove('games/chess')            # Delete a game
>>> g.size                              # File count is automatically updated
2
```
除了说明描述器如何运行计算，这个例子也揭示了 `__get__()` 参数的目的。形参 *self* 接收的实参是 \_size\_，即 *DirectorySize* 的一个实例。形参 *obj* 接收的实参是 *g* 或 \_s\_，即 *Directory* 的一个实例。而正是 *obj* 让 `__get__()` 方法获得了作为目标的目录。形参 *objtype* 接收的实参是 *Directory* 类