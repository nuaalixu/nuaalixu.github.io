---
layout: post
category: Python编程
---

# String Formatting Best Practices
## Introduction
介绍四种字符串格式化的方式：

1. “Old Style“ (% Operator)
2. “New Style” (str.format)
3. f-Strings (Python 3.6+)
4. Template Strings (Standard Library)

如何选择？

![image](/images/U0CvWQB8C4S4gK2UrbF43td9lWwPu8sbmyGX8qQDJtE.png)



## “Old Style”-不推荐
利用`%`操作符配合说明符，进行C风格的字符串格式化。

```python
>>> 'Hello, %s' % name
"Hello, Bob"
```
Python3后该风格的格式化被新方式替代，不推荐使用。

## “New Style”
通过`str.format()`的方式进行字符串格式化。

```python
>>> 'Hello, {}'.format(name)
'Hello, Bob'
```
也可以根据名称指定位置。

```python
>>> 'Hey {name}, there is a 0x{errno:x} error!'.format(
...     name=name, errno=errno)
'Hey Bob, there is a 0xbadc0ffee error!'
```
说明符以`:x` 的形式指定。

优势是可以将格式化字符串和数值分开定义。

## f-Strings
又称作格式化字符串字面量，以`f` 前缀作为标志。

```python
>>> f'Hello, {name}!'
'Hello, Bob!'
```
f-string的一个优势是能够嵌入任意的python表达式。

## Template Strings
Template Strings 由`string`标准库提供，它的功能没有上述的格式化方式强大，但是更安全，更适合格式化来自外部的字符串。

例如，要处理来自用户输入的字符串：

```python
>>> # This is our super secret key:
>>> SECRET = 'this-is-a-secret'

>>> class Error:
...      def __init__(self):
...          pass

>>> # A malicious user can craft a format string that
>>> # can read data from the global namespace:
>>> user_input = '{error.__init__.__globals__[SECRET]}'

>>> # This allows them to exfiltrate sensitive information,
>>> # like the secret key:
>>> err = Error()
>>> user_input.format(error=err)
'this-is-a-secret'
```
上述代码，`SECRET`被泄漏，使用Template可以避免该风险。

```python
>>> user_input = '${error.__init__.__globals__[SECRET]}'
>>> Template(user_input).substitute(error=err)
ValueError:
"Invalid placeholder in string: line 1, col 1"
```
此外Template Strings不支持格式说明符，需要先转换再替换。

