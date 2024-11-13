---
layout: post
category: Python编程
---

# 元类 metaclass
“元类”是类的类，类是元类的实例。
python中“一切皆对象“，对象是类的实例化，类本身也是元类的实例。
元类用来控制类的创建，正如类用来控制对象的创建。
type是python中基础的元类，是自定义类默认的元类。

比喻：儿子（子类）继承了父亲（超类），儿子和父亲都属于同一个维度的生物，儿子的行为大多和父亲相似（继承和覆盖），他们都由上帝（元类）创造。

## 类的创建
```python
class classname:
    pass
```
class classname 指实例化一个元类的对象，默认的元类是type，相当于调用type(classname, superclass, attributedict)，实例化出一个用户自定义的类class。

## 类的实例化
```python
foo = classname()
```
类的实例化时，python内部调用了classname的元类（此处为type）的 `__call__` 方法，type的默认`__call__` 方法会依次调用 `__new__` 方法和 `__init__` 方法。其中 `__new__` 方法分配内存，`__init__` 方法初始化对象。
如果 classname 中重写了 `__new__` 和 `__init__` 方法，那么type的 `__call__` 方法实际调用的是 classname 的方法，否则是type的默认方法。
可以通过重写 classname 的 `__new__` 和 `__init__` 方法，控制实例化 classname 时的操作。

## 自定义元类
```python
class meta(type):
    pass
```
通过继承基类type，来自定义元类。

```python
class classname(metaclass=meta):
    pass
```
class classname 指实例化一个元类的对象，此时指定元类是meta，相当于调用meta(classname, superclass, attributedict)进行实例化。和class实例化一个对象类似，底层会调用meta的 `__new__` 和 `__init__` 方法，可以重写该方法，控制定义 classname 时的操作。

```python
foo = classname()
```
已知一个类实现了`__call__`方法后，它的实例支持以圆括号的形式被调用。

`classname()` 中，`classname`是元类`type`的实例，`classname()`这个形式是在调用`type`中定义的`__call__` 方法。

所以类的实例化，调用了元类的 `__call__` 方法，元类默认的 `__call__` 方法又调用了 `__new__` 和 `__init__` 方法，可重写该方法，以自定义类的实例化。

## 例子
```python
In[15]: class Mymeta(type):
   ...:     def __init__(self, name, bases, dic):
   ...:         super().__init__(name, bases, dic)
   ...:         print('===>Mymeta.__init__')
   ...:         print(self.__name__)
   ...:         print(dic)
   ...:         print(self.yaml_tag)
   ...: 
   ...:     def __new__(cls, *args, **kwargs):
   ...:         print('===>Mymeta.__new__')
   ...:         print(cls.__name__)
   ...:         return type.__new__(cls, *args, **kwargs)
   ...: 
   ...:     def __call__(cls, *args, **kwargs):
   ...:         print('===>Mymeta.__call__')
   ...:         obj = cls.__new__(cls)
   ...:         cls.__init__(cls, *args, **kwargs)
   ...:         return obj
   ...: 
In[16]: 
In[16]: class Foo(metaclass=Mymeta):
   ...:     yaml_tag = '!Foo'
   ...: 
   ...:     def __init__(self, name):
   ...:         print('Foo.__init__')
   ...:         self.name = name
   ...: 
   ...:     def __new__(cls, *args, **kwargs):
   ...:         print('Foo.__new__')
   ...:         return object.__new__(cls)
   ...:     
===>Mymeta.__new__
Mymeta
===>Mymeta.__init__
Foo
{'__module__': '__main__', '__qualname__': 'Foo', 'yaml_tag': '!Foo', '__init__': <function Foo.__init__ at 0x0000000007EF3828>, '__new__': <function Foo.__new__ at 0x0000000007EF3558>}
!Foo

In[17]: foo = Foo('foo')
===>Mymeta.__call__
Foo.__new__
Foo.__init__

In[18]:
```
从上面的运行结果可以发现在定义 class Foo() 定义时，会依次调用 MyMeta 的 `__new__`  和 `__init__` 方法构建 Foo 类，然后在调用 foo = Foo() 创建类的实例对象时，才会调用 MyMeta 的 `__call__` 方法来调用 Foo 类的 `__new__`  和 `__init__` 方法。