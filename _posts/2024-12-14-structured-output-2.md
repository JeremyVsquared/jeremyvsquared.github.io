---
layout: post
title: Revisiting structured output
date: 2024-12-14
description: Structured output updates for Ollama library
---

[As announced in a blog post](https://ollama.com/blog/structured-outputs), Ollama was recently updated with built in support for structured output. This can be used in conjunction with pydantic class definitions. This should generally function reliably when provided as the `format` parameter to the `chat()` function, but it can be enhanced by adding "_return as json_" to the prompt.


```python
from ollama import chat
from pydantic import BaseModel
```

# Country example


```python
class Country(BaseModel):
    name: str
    capital: str
    language: list[str]
    population: int
```


```python
response = chat(
    messages=[{
        'role': 'user',
        'content': 'tell me about UAE'
    }],
    model='llama3.1',
    format=Country.model_json_schema()
)
```


```python
print(response)
print()
print(response.message.content)
```

    model='llama3.1' created_at='2025-01-07T23:30:16.72438Z' done=True done_reason='stop' total_duration=21199967041 load_duration=11874474708 prompt_eval_count=14 prompt_eval_duration=6948000000 eval_count=39 eval_duration=2373000000 message=Message(role='assistant', content='{ "name": "United Arab Emirates" , "capital": "Abu Dhabi", "language": ["Arabic","English"], "population": 9 }\n  \t\t\t\t\t\t\t \t', images=None, tool_calls=None)
    
    { "name": "United Arab Emirates" , "capital": "Abu Dhabi", "language": ["Arabic","English"], "population": 9 }
      							 	


# Weather example

Given a tool to search the web for current weather conditions would make this much more helpful, but the purpose here is to work with structured output.


```python
class Weather(BaseModel):
    temperature: float
    humidity: int
    weather_condition: str
```

Let's examine the output of `model_json_schema()`


```python
output_json_schema = Weather.model_json_schema()

output_json_schema
```




    {'properties': {'temperature': {'title': 'Temperature', 'type': 'number'},
      'humidity': {'title': 'Humidity', 'type': 'integer'},
      'weather_condition': {'title': 'Weather Condition', 'type': 'string'}},
     'required': ['temperature', 'humidity', 'weather_condition'],
     'title': 'Weather',
     'type': 'object'}




```python
response = chat(
    messages=[{
        "role": 'user',
        'content': 'what is the current weather in new york?'
    }],
    model='llama3.1',
    format=output_json_schema
)
```


```python
print(response)
print()
print(response.message.content)
```

    model='llama3.1' created_at='2025-01-07T23:33:58.534662Z' done=True done_reason='stop' total_duration=3735988417 load_duration=29947292 prompt_eval_count=19 prompt_eval_duration=2039000000 eval_count=27 eval_duration=1664000000 message=Message(role='assistant', content='{ "temperature": 55.1, "humidity": 71, "weather_condition": "Mostly Cloudy"}', images=None, tool_calls=None)
    
    { "temperature": 55.1, "humidity": 71, "weather_condition": "Mostly Cloudy"}


# Traffic light

Here we would presume to use a vision model with an image input with the intent of detecting the status of a traffic light.


```python
class TrafficLightStatus(BaseModel):
    red: bool
    yellow: bool
    green: bool
```


```python
output_json_schmea = TrafficLightStatus.model_json_schema()
```


```python
with open('traffic_light.jpg', 'rb') as img:
    img_data = img.read()
```


```python
response = chat(
    messages=[{
        "role": 'user',
        'content': """examine the given image and provide the traffic light status if there is a traffic light in the image. 
                      if no traffic lights are detected in the image, simply return false for all traffic light colors.""",
        'images': [img_data]
    }],
    model='llama3.2-vision',
    format=output_json_schema
)
```


```python

```
