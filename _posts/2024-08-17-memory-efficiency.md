---
title: Data structure memory efficiency
date: 2024-08-17
---

Looking at data structure memory and computational efficiency

- dict
- namedtuples
- typeddict
- pydantic


```python
import sys
```

# `dict`

A dictionary in Python is a mutable, unordered collection of key-value pairs. It's versatile but can be memory-intensive for small collections.


```python
# Dictionary
def dict_example():
    data = {'name': 'Alice', 'age': 25, 'height': 170.5}
    return sys.getsizeof(data)
```


```python
print(f"dict: {dict_example()} bytes")
```

    dict: 232 bytes


# `collections.namedtuple`

Named tuples provide a lightweight way to create tuple subclasses with named fields, offering more readability than regular tuples with similar memory efficiency.


```python
from collections import namedtuple
```


```python
# Named Tuple
Person = namedtuple('Person', ['name', 'age', 'height'])
def namedtuple_example():
    data = Person('Alice', 25, 170.5)
    return sys.getsizeof(data)
```


```python
print(f"namedtuple: {namedtuple_example()} bytes")
```

    namedtuple: 64 bytes


# `typing.TypedDict`

TypedDict is used for type-checking dictionaries where keys have specific types. However, at runtime, it's just a dict, so memory usage is similar to dict.


```python
from typing import TypedDict
```


```python
# TypedDict
class PersonTyped(TypedDict):
    name: str
    age: int
    height: float

def typeddict_example():
    data = PersonTyped(name='Alice', age=25, height=170.5)
    # Note: TypedDict is a type hint, so we need to use a regular dict for size measurement
    return sys.getsizeof(dict(data))
```


```python
print(f"TypedDict: {typeddict_example()} bytes")
```

    TypedDict: 232 bytes


# `dataclass`

Dataclasses provide a way to automatically add generated special methods like __init__() and __repr__() to classes, with less boilerplate code.



```python
from dataclasses import dataclass
```


```python
@dataclass
class PersonData:
    name: str
    age: int
    height: float

def dataclass_example():
    data = PersonData(name="Alice", age=25, height=170.5)
    return sys.getsizeof(data)
```


```python
print(f"dataclass: {dataclass_example()} bytes")

```

    dataclass: 48 bytes


# `pydantic.BaseModel`

Pydantic models provide runtime data validation with strong typing, but this comes with additional memory overhead due to validation mechanisms and metadata.


```python
from pydantic import BaseModel
```


```python
# Pydantic Model
class PersonPydantic(BaseModel):
    name: str
    age: int
    height: float

def pydantic_example():
    data = PersonPydantic(name='Alice', age=25, height=170.5)
    return sys.getsizeof(data)

def pydantic_modeldump_example():
    data = PersonPydantic(name='Alice', age=25, height=170.5)
    return sys.getsizeof(data.model_dump())
```


```python
print(f"pydantic: {pydantic_example()} bytes")
print(f"pydantic (model_dump()): {pydantic_modeldump_example()}")
```

    pydantic: 72 bytes
    pydantic (model_dump()): 232


# Comparing all types

Reviewing all model memory usage


```python
# Comparison
structures = [
    ('dict', dict_example),
    ('namedtuple', namedtuple_example),
    ('TypedDict', typeddict_example),
    ('Pydantic', pydantic_example),
    ('Pydantic (modeldump())', pydantic_modeldump_example)
]

print("Memory Usage Comparison:")
for name, func in structures:
    size = func()
    print(f"{name}: {size} bytes")
```

    Memory Usage Comparison:
    dict: 232 bytes
    namedtuple: 64 bytes
    TypedDict: 232 bytes
    Pydantic: 72 bytes
    Pydantic (modeldump()): 232 bytes



```python

```
