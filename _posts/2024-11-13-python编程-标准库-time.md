---
layout: post
category: Python编程
---

# 标准库: time
## 预备知识
真太阳时：观测太阳的在天空中的位置，不均匀

平太阳时：假想太阳均匀移动

格林尼治平时：格林尼治天文台所在地的地方平太阳时，地球自转仍不均匀

世界时：明确表示每天从午夜开始的格林尼治平时。

国际原子时（TAI）：使用原子钟计时

协调世界时（UTC）：使定义的时间与地球自转相配合，人们通过在TAI的基础上不定期增减闰秒的方式，使定义的时间与世界时保持差异在0.9秒以内

格林尼治标准时间：“格林尼治平时”一词在民用领域常常被认为与UTC相同，其“平时”（平太阳时）的含义已部分缺失，所以在中文里又改称“格林尼治标准时间”。

总结，UTC是最主要的世界时间标准，其以原子时的秒长为基础，在时刻上尽量接近于格林尼治标准时间。

## time function
### time.time() -> float
返回1970年1月1日0点0分0秒（UTC）至今经过的时间（秒），带小数点。

### time.gmtime(\[secs\]) -> time\_struct
将经过的时间秒转换为UTC（等同于格林尼治标准时）表示，默认参数是time.time()返回值。

### time.localtime(\[secs\]) -> time\_struct
将经过的时间秒转换为当地时间表示，默认参数是time.time()返回值。

### time.asctime(\[tuple\]) -> str
将时间转换为字符串，默认参数是time.localtime()返回值。

### time.mktime(`time_struct`) -> float
将localtime结构体转换为从epoch开始的秒。内部是根据时区转为utc，然后和epoch做比较，计算时间。

### time.strptime & time.strftime
时间字符串和`time.struct_time` 的互转，该结构相当于一个`namedtuple` 。

前者字符串到`time.struct_time`

```python
date_string = "2022-01-25"
format_string = "%Y-%m-%d"

parsed_time = time.strptime(date_string, format_string)

print(parsed_time)
# time.struct_time(tm_year=2022, tm_mon=1, tm_mday=25, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=25, tm_isdst=-1)
```
后者`time.struct_time` 到字符串

```python
current_time = time.localtime()
format_string = "%Y-%m-%d %H:%M:%S"

formatted_time = time.strftime(format_string, current_time)

print(formatted_time)
# 2024-03-27 19:08:20
```
时间格式化字符串：

1. 完整日期和时间:

* `"%Y-%m-%d %H:%M:%S"`: 2023-06-22 15:30:45
* `"%m/%d/%Y %H:%M:%S"`: 06/22/2023 15:30:45
* `"%b %d %Y %H:%M:%S"`: Jun 22 2023 15:30:452. 
2. 仅日期:
* `"%Y-%m-%d"`: 2023-06-22
* `"%m/%d/%Y"`: 06/22/2023
* `"%b %d %Y"`: Jun 22 2023
3. 仅时间:
* `"%H:%M:%S"`: 15:30:45
* `"%I:%M:%S %p"`: 03:30:45 PM
4. 缩写的月份名称:
* `"%b"`: Jun
* `"%a"`: Thu (表示星期)
5.  数字表示的月份和星期:
* `"%m"`: 06 (表示月份)
* `"%w"`: 4 (表示星期,0表示星期日)

## datetime标准库
datetime 模块提供用于操作日期和时间的类。 虽然支持日期和时间算术，但实现的重点是用于输出格式化和操作的有效属性提取。

### `time.struct_time` 和`datetime.datetime`的互转
```python
# structTime is ready
datetime.datetime(*structTime[:6])
datetime.datetime(2009, 11, 8, 20, 32, 35)
```
### float到`datetime.datetime`
`date.fromtimestamp(`*timestamp*`)` 可以将`time.time()`的float秒转换成`datatime` 对象。

### datetime.strptime & datetime.strftime
字符串和`datatime.datetime` 对象的互转。同`time` 模块。

## Wall time vs CPU time
二者区别：

* wall time 是开始计时到结束计时之间的全部时长，包括等待资源的时间。
* cpu time 只包含cpu执行指令的时间，不包括等待资源的时间（比如I/O时间）。

`time.time()` 之间的差值是wall time，和某个进程无关。

`time.process_time()`之间的差值是该进程的cpu time，即实际运行在cpu上的时间，注意参考点未知，所以两次调用之间的值才有意义。

`time.perf_counter()`之间的差值是某个程序代码的执行时间，包括sleep时间。