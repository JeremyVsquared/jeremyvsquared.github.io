---
title: Reliability of structured output from small models
date: 2024/10/05
---

# Reliability of structured output from small models

We will explore consistently properly formatted structured output from smaller models. Specifically we're going to work with the `llama3.2` model and compare these variants to the `llama3.1:8b` model (4.9 gig). At 2 gig for the 3b and 1.3 gig for the 1b models, it is a very memory efficient model to work with but can suffer from reliability issues common to smaller model.

We will base the tests on asking the LLM's to answer a simple question in the form of valid JSON output. We are not interested in the LLM's answering the correctly, only that they follow formatting instructions for the output based mostly upon the prompting alone.


```python
import time, json, random
from langchain_ollama import ChatOllama
```

Make sure Ollama is running


```python
chat = ChatOllama(model="llama3.2:1b")
print(chat.invoke("hey there"))
```

    content='Hello again. How can I assist you today?' additional_kwargs={} response_metadata={'model': 'llama3.2:1b', 'created_at': '2024-12-08T23:26:19.25622518Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 125630910, 'load_duration': 17527930, 'prompt_eval_count': 27, 'prompt_eval_duration': 22931000, 'eval_count': 11, 'eval_duration': 42493000} id='run-728e4f05-1501-4ce8-a2e6-21e78c7a1298-0' usage_metadata={'input_tokens': 27, 'output_tokens': 11, 'total_tokens': 38}


Create a simple test function to make evaluating different prompts easier


```python
def test_prompt(prompt, option_list=[], iterations=1000, models=["llama3.2:1b", "llama3.2", "llama3.1"]):
    # Initialize an empty dictionary to store the responses and statistics
    results = {}

    for model in models:
        print("running {} iterations for {}".format(iterations, model))
        start_time = time.time()
        
        # Initialize lists for storing successes, failures, and responses
        successes = 0
        failures = 0
        responses = []
        
        for i in range(iterations):
            chat = ChatOllama(model=model)

            format_input = option_list[random.randint(0, len(option_list)-1)]

            rsp = chat.invoke(prompt.format(format_input))
            
            responses.append(rsp)  # Store the response in the list
            
            try:
                json.loads(rsp.replace('```json', '').replace('```', ''))
                successes += 1  # Store the iteration number if successful
            except:
                failures += 1  # Store the iteration number if failed
        
        stop_time = time.time()

        # Store the results in the dictionary using the model name as the key
        results[model] = {
            "successes": successes,
            "failures": failures,
            "responses": responses,
            "total_time": stop_time - start_time,
            "failure_rate": (failures / iterations) * 100.0
        }

    print("\nResults")
    print("-"*40)
    # Optionally, print the results if you want to see the result
    for model, data in results.items():
        print(model)
        print("\tSuccesses:", data['successes'])
        print("\tFailures:", data['failures'])
        print(f"\tFailure rate: {data['failure_rate']:.2f}%")
        print(f"\tTotal time: {data['total_time']:.2f} seconds")

        print(f"\tAverage time per request: {data['total_time']/len(data['responses']):.2f} seconds")

    return results
```


    Failed to connect to the remote Jupyter Server 'http://10.69.8.75:8888/'. Verify the server is running and reachable. (Failed to connect to the remote Jupyter Server 'http://10.69.8.75:8888/'. Verify the server is running and reachable. (request to http://10.69.8.75:8888/api/kernels?1737660141481 failed, reason: connect ETIMEDOUT 10.69.8.75:8888).).



```python
def is_failure(rsp):
    try:
        json.loads(rsp.replace('```json', '').replace('```', ''))
        return False
    except:
        return True
```


```python
iterations_per_eval = 1000
```

## 1. Simple prompt

We'll start with a very simple, straight forward request for JSON output to establish a baseline performance. The llama models will occasionally output JSON with markdown code quotes so we'll give it a handicap and remove those from the output.


```python
prompt_simple = """Please tell me if the following creature is a mammal: {}.\n
            Answer in JSON format with a single key of 'is_mammal' and a value of 
            'true' if it is a mammal or 'false' if it is not a mammal."""
creatures = ['monkey', 'horse', 'lizard', 'falcon', 'slug', 'dog', 
             'octopus', 'spider', 'frog', 'leopard', 'automobile',
             ' ', 'ladder', '4 bedroom, 3 bath']
```


```python
results = test_prompt(prompt_simple, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 924
    	Failures: 76
    	Failure rate: 7.60%
    	Total time: 211.43 seconds
    	Average time per request: 0.21 seconds
    llama3.2
    	Successes: 902
    	Failures: 98
    	Failure rate: 9.80%
    	Total time: 237.23 seconds
    	Average time per request: 0.24 seconds
    llama3.1
    	Successes: 870
    	Failures: 130
    	Failure rate: 13.00%
    	Total time: 401.65 seconds
    	Average time per request: 0.40 seconds


MacBook Air M2 16GB results:

```
running 1000 iterations for llama3.2:1b
running 1000 iterations for llama3.2
running 1000 iterations for llama3.1

Results
----------------------------------------
llama3.2:1b
	Successes: 889
	Failures: 111
	Failure rate: 11.10%
	Total time: 369.26 seconds
	Average time per request: 0.37 seconds
llama3.2
	Successes: 871
	Failures: 129
	Failure rate: 12.90%
	Total time: 777.81 seconds
	Average time per request: 0.78 seconds
llama3.1
	Successes: 874
	Failures: 126
	Failure rate: 12.60%
	Total time: 2372.85 seconds
	Average time per request: 2.37 seconds
```


```python
print("Llama 3.1 failure sample:")
print("\n".join([i for i in results['llama3.1']['responses'] if is_failure(i)][:5]))
print("-"*80)
print("Llama 3.2:1b failure sample:")
print("\n".join([i for i in results['llama3.2:1b']['responses'] if is_failure(i)][:5]))
print("-"*80)
print("Llama 3.2:3b failure sample:")
print("\n".join([i for i in results['llama3.2']['responses'] if is_failure(i)][:5]))

```

    Llama 3.1 failure sample:
    ```json
    {
        "is_mammal": false
    }
    ```
    
    This is because frogs are amphibians, not mammals. They belong to the class Amphibia, which also includes toads, salamanders, and newts.
    ```json
    {
        "is_mammal": false
    }
    ```
    
    This creature, a 4 bedroom, 3 bath house, does not have characteristics associated with mammals (suckling young, fur/hair, etc.) and is therefore classified as an inanimate object.
    I don't see any information about the creature you're asking me to classify as a mammal.
    
    Please provide more details or describe the creature, and I'll be happy to help! 
    
    (If you meant this literally, I'd love some kind of hint)
    ```json
    {
      "is_mammal": false
    }
    ```
    
    This is because falcons are birds, not mammals.
    However, I don't see any description of the creature. Could you please provide more information about the creature you're asking about? 
    
    If you'd like to describe the creature and ask me if it's a mammal, I'll be happy to help!
    --------------------------------------------------------------------------------
    Llama 3.2:1b failure sample:
    {
        "is_mammal": false
    {
      "is_mammal": false
    Unfortunately, I don't see any information about the creature you're referring to. Can you please provide more context or details about this creature, such as its appearance, habitat, or any other characteristics? This will help me give you a more accurate answer in JSON format.
    
    Once I have the necessary information, I'll be happy to tell you whether it is a mammal or not.
    Here's the answer in JSON format:
    
    ```json
    {
        "is_mammal": false
    }
    ```
    ## Mammal Status of Given Creature
    
    ```json
    {
      "is_mammal": false
    }
    ```
    --------------------------------------------------------------------------------
    Llama 3.2:3b failure sample:
    I don't see the creature you're referring to. Please provide the name or description of the creature, and I'll be happy to help in JSON format.
    I'd be happy to help, but I don't see a description of the creature you're referring to. Could you please provide more information about the creature, such as its characteristics, features, or classification? Once I have this information, I can provide an answer in JSON format.
    
    For example, if you describe it as "a large carnivorous predator with claws and a tail", I could then tell you whether it is a mammal or not.
    I don't see any creature mentioned in your question. Could you please provide the name or description of the creature, and I'll be happy to help?
    {'is_mammal': false}
    I don't see any information about the creature you are referring to. Could you please provide more details or clarify which creature you would like me to determine?


## 2. Counter prompting

Many failures appear to be due to the LLM adding unnecessary explanation to the output. Let's try to counter this by telling it explicitly not to do so. We will also include instruction to not use markdown formatting.


```python
prompt_counter = """Please tell me if the following creature is a mammal: {}.
                    \n
                    Answer in JSON format with a single key of 'is_mammal' and a value of 
                    'true' if it is a mammal or 'false' if it is not a mammal.
                    \n
                    Do not include a preamble or explanation of the answer,
                    do not format the JSON output as markdown text,
                    only return properly formatted JSON."""
creatures = ['monkey', 'horse', 'lizard', 'falcon', 'slug', 'dog', 
             'octopus', 'spider', 'frog', 'leopard', 'automobile',
             ' ', 'ladder', '4 bedroom, 3 bath']
```


```python
results = test_prompt(prompt_counter, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 995
    	Failures: 5
    	Failure rate: 0.50%
    	Total time: 201.28 seconds
    	Average time per request: 0.20 seconds
    llama3.2
    	Successes: 1000
    	Failures: 0
    	Failure rate: 0.00%
    	Total time: 219.95 seconds
    	Average time per request: 0.22 seconds
    llama3.1
    	Successes: 1000
    	Failures: 0
    	Failure rate: 0.00%
    	Total time: 270.96 seconds
    	Average time per request: 0.27 seconds


MacBook Air M2 16GB results:

```
running 1000 iterations for llama3.2:1b
running 1000 iterations for llama3.2
running 1000 iterations for llama3.1

Results
----------------------------------------
llama3.2:1b
	Successes: 990
	Failures: 10
	Failure rate: 1.00%
	Total time: 356.12 seconds
	Average time per request: 0.36 seconds
llama3.2
	Successes: 1000
	Failures: 0
	Failure rate: 0.00%
	Total time: 722.69 seconds
	Average time per request: 0.72 seconds
llama3.1
	Successes: 1000
	Failures: 0
	Failure rate: 0.00%
	Total time: 1378.54 seconds
	Average time per request: 1.38 seconds
```

The output has greatly improved with a simple counter prompt. Even our smallest and least capable model produced a failure rate of 1%.


```python
print("Llama 3.1 response sample:")
print("\n".join(results['llama3.1']['responses'][:5]))
print("-"*80)
print("Llama 3.2:1b response sample:")
print("\n".join(results['llama3.2:1b']['responses'][:5]))
print("-"*80)
print("Llama 3.2:3b response sample:")
print("\n".join(results['llama3.2']['responses'][:5]))
```

    Llama 3.1 response sample:
    {"is_mammal": false}
    {"is_mammal": false}
    {"is_mammal": false}
    {"is_mammal": true}
    {
      "is_mammal": false
    }
    --------------------------------------------------------------------------------
    Llama 3.2:1b response sample:
    {"is_mammal": "false"}
    {"is_mammal": true}
    {
      "is_mammal": false
    }
    {
      "is_mammal": true
    }
    {
      "is_mammal": false
    }
    --------------------------------------------------------------------------------
    Llama 3.2:3b response sample:
    {"is_mammal": true}
    {"is_mammal": false}
    {"is_mammal": "false"}
    {"is_mammal": "false"}
    {"is_mammal": false}


## 3. Few shot

The performance has significantly improved with counter prompting, but let's add a few examples to the prompt to see how that performs. Even in the case of 0 JSON validation errors, the output is inconsistent with quotes and line breaks. Ideally we'd like to see a 0% failure rate as well as consistent, clean output formatting. 


```python
prompt_fewshot = """Please tell me if the following creature is a mammal: {}.\n
                    Answer in JSON format with a single key of 'is_mammal' and a value of 
                    'true' if it is a mammal or 'false' if it is not a mammal.
                    \n
                    Do not include a preamble or explanation of the answer,
                    do not format the JSON output as markdown text,
                    only return properly formatted JSON.
                    \n\n
                    For example, when asked if an "ocelot" is a mammal, your response
                    should be:\n
                    {{"is_mammal": "true"}}
                    \n\n
                    When asked if a "velociraptor" is a mammal, your response should be:\n
                    {{"is_mammal": "false"}}
                    \n\n
                    When asked if a "mid size sedan" is a mammal, your response should be:\n
                    {{"is_mammal": "false"}}"""
creatures = ['monkey', 'horse', 'lizard', 'falcon', 'slug', 'dog', 
             'octopus', 'spider', 'frog', 'leopard', 'automobile',
             ' ', 'ladder', '4 bedroom, 3 bath']
```


```python
results = test_prompt(prompt_fewshot, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 973
    	Failures: 27
    	Failure rate: 2.70%
    	Total time: 200.16 seconds
    	Average time per request: 0.20 seconds
    llama3.2
    	Successes: 998
    	Failures: 2
    	Failure rate: 0.20%
    	Total time: 233.29 seconds
    	Average time per request: 0.23 seconds
    llama3.1
    	Successes: 1000
    	Failures: 0
    	Failure rate: 0.00%
    	Total time: 289.05 seconds
    	Average time per request: 0.29 seconds


MacBook Air M2 16GB results:

```
running 1000 iterations for llama3.2:1b
running 1000 iterations for llama3.2
running 1000 iterations for llama3.1

Results
----------------------------------------
llama3.2:1b
	Successes: 976
	Failures: 24
	Failure rate: 2.40%
	Total time: 701.14 seconds
	Average time per request: 0.70 seconds
llama3.2
	Successes: 996
	Failures: 4
	Failure rate: 0.40%
	Total time: 1383.39 seconds
	Average time per request: 1.38 seconds
llama3.1
	Successes: 1000
	Failures: 0
	Failure rate: 0.00%
	Total time: 3046.14 seconds
	Average time per request: 3.05 seconds
```


```python
print("Llama 3.1 response sample:")
print("\n".join(results['llama3.1']['responses'][:5]))
print("-"*80)
print("Llama 3.2:1b response sample:")
print("\n".join(results['llama3.2:1b']['responses'][:5]))
print("-"*80)
print("Llama 3.2:3b response sample:")
print("\n".join(results['llama3.2']['responses'][:5]))
```

    Llama 3.1 response sample:
    {"is_mammal": "true"}
    {"is_mammal": "false"}
    {"is_mammal": "false"}
    {"is_mammal": "false"}
    {"is_mammal": "true"}
    --------------------------------------------------------------------------------
    Llama 3.2:1b response sample:
    {"is_mammal": "true"}
    {"is_mammal": false}
    {"is_mammal": "false"}
    {"is_mammal": "false"}
    {"is_mammal": "false"}
    --------------------------------------------------------------------------------
    Llama 3.2:3b response sample:
    {"is_mammal": "false"}
    {"is_mammal": "true"}
    {"is_mammal": "true"}
    {"is_mammal": "false"}
    {"is_mammal": "true"}



```python
print("Llama 3.1 failure sample:")
print("\n".join([i for i in results['llama3.1']['responses'] if is_failure(i)][:4]))
print("-"*80)
print("Llama 3.2:1b failure sample:")
print("\n".join([i for i in results['llama3.2:1b']['responses'] if is_failure(i)][:4]))
print("-"*80)
print("Llama 3.2:3b failure sample:")
print("\n".join([i for i in results['llama3.2']['responses'] if is_failure(i)][:4]))
```

    Llama 3.1 failure sample:
    
    --------------------------------------------------------------------------------
    Llama 3.2:1b failure sample:
    {"is_mammal": "false}"
    {"is_mammal": "false}"
    {"is_mammal": "false}"
    {"is_mammal": "false}"
    --------------------------------------------------------------------------------
    Llama 3.2:3b failure sample:
    {"is_mammal": "true"}
    
    {"is_mammal": "false"}
    
    {"is_mammal": "false"}
    {"is_mammal": "true"}
    
    {"is_mammal": "false"}
    
    {"is_mammal": "false"}


This output is looking much better. While the counter prompting elliminated JSON parsing errors, this output is looking much cleaner. The `llama3.2:1b` model does appear to struggle more with simple, clean output but it is parsed as valid JSON.

## 4. Langchain output parsing

Now we will evaluate using the langchain json output parsing and formatting instructions. We will start with the simple prompt in order to get the most clear example of the efficacy of this method without extra help. 

The langchain JSON output parser will throw an exception when the generation is not valid JSON so we will alter the testing function to catch these exceptions and add them to our `failures` count.


```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
```


```python
class IsMammal(BaseModel):
    is_mammal: bool= Field(description="Boolean value answering whether or not the given animal is a mammal")

parser = JsonOutputParser(pydantic_object=IsMammal)
```


```python
def test_prompt_template(prompt, option_list=[], iterations=1000, models=["llama3.2:1b", "llama3.2", "llama3.1"]):
    # Initialize an empty dictionary to store the responses and statistics
    results = {}

    for model in models:
        print("running {} iterations for {}".format(iterations, model))
        start_time = time.time()
        
        # Initialize lists for storing successes, failures, and responses
        successes = 0
        failures = 0
        responses = []
        
        for i in range(iterations):
            prompt_template = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{query}",
                input_variables=['query'],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            chat = ChatOllama(model=model)

            format_input = option_list[random.randint(0, len(option_list)-1)]

            chain = prompt_template | chat | parser
            
            try:
                rsp = chain.invoke({"query": prompt.format(format_input)})
                responses.append(rsp)  # Store the response in the list
            
                if 'is_mammal' in rsp:
                    successes += 1  # Store the iteration number if successful
                else:
                    failures += 1
            except Exception as e:
                responses.append(str(e))
                failures += 1  # Store the iteration number if failed
                
            
        
        stop_time = time.time()

        # Store the results in the dictionary using the model name as the key
        results[model] = {
            "successes": successes,
            "failures": failures,
            "responses": responses,
            "total_time": stop_time - start_time,
            "failure_rate": (failures / iterations) * 100.0
        }

    print("\nResults")
    print("-"*40)
    # Optionally, print the results if you want to see the result
    for model, data in results.items():
        print(model)
        print("\tSuccesses:", data['successes'])
        print("\tFailures:", data['failures'])
        print(f"\tFailure rate: {data['failure_rate']:.2f}%")
        print(f"\tTotal time: {data['total_time']:.2f} seconds")

        print(f"\tAverage time per request: {data['total_time']/len(data['responses']):.2f} seconds")

    return results
```

### Plain prompt


```python
prompt_plain = """Please tell me if the following creature is a mammal: {}."""
```


```python
results = test_prompt_template(prompt_plain, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 257
    	Failures: 743
    	Failure rate: 74.30%
    	Total time: 781.17 seconds
    	Average time per request: 0.77 seconds
    llama3.2
    	Successes: 134
    	Failures: 866
    	Failure rate: 86.60%
    	Total time: 687.56 seconds
    	Average time per request: 0.69 seconds
    llama3.1
    	Successes: 998
    	Failures: 2
    	Failure rate: 0.20%
    	Total time: 283.21 seconds
    	Average time per request: 0.28 seconds



```python
print("Llama 3.1 response sample:")
print(results['llama3.1']['responses'][:5])
print("-"*80)
print("Llama 3.2:1b response sample:")
print(results['llama3.2:1b']['responses'][:5])
print("-"*80)
print("Llama 3.2:3b response sample:")
print(results['llama3.2']['responses'][:5])
```

    Llama 3.1 response sample:
    [{'is_mammal': True}, {'is_mammal': False}, {'is_mammal': False}, {'is_mammal': False}, {'is_mammal': False}]
    --------------------------------------------------------------------------------
    Llama 3.2:1b response sample:
    ['Invalid json output: I can help you determine if the slug is a mammal.\n\nHere\'s the relevant information from the schema:\n\n* The property "slug" has a value of "slug".\n* According to the schema, this property should be classified as a type (in this case, "type": "object", but it seems like an array was used instead) and contain an array with two items.\n* However, in your question, you asked me about the slug\'s classification as a mammal.\n\nUnfortunately, I couldn\'t find any information on a creature called a "slug" that is classified as a mammal. Slugs are actually mollusks, which are a separate class of animals. Mollusks include snails, slugs, oysters, and clams, among others.\n\nSo, to answer your question, the slug is not a mammal.', {'foo': [{'title': 'is_mammal', 'description': 'Boolean value answering whether or not the given animal is a mammal', 'type': 'boolean'}]}, {'is_mammal': False}, 'Invalid json output: I\'m happy to help you with your query.\n\nTo determine if the creature "lizard" is a mammal, I\'ll need to know more about what a mammal is. Is it a living organism that breathes air and has hair or fur? Or is there another definition of a mammal that applies?\n\nAssuming that a lizard is not considered a mammal (and this might be the intended definition), I can provide you with information on whether or not "lizard" meets the criteria for being a mammal.\n\nSince lizards do have scales and are cold-blooded, but they also have hair, lungs, and a four-chambered heart, I would say that "lizard" is not a mammal. However, this classification can vary depending on the specific type of lizard and its characteristics.\n\nIf you\'d like to know more about lizards or if there\'s another definition of a mammal you\'re interested in, feel free to let me know!', {'is_mammal': True}]
    --------------------------------------------------------------------------------
    Llama 3.2:3b response sample:
    ['Invalid json output: This will return a JSON object indicating that the ladder is not a mammal.\n\n{"is_mammal": false}', {'properties': {'is_mammal': {'description': 'Boolean value answering whether or not the given animal is a mammal', 'title': 'Is Mammal', 'type': 'boolean'}}, 'required': ['is_mammal']}, {'properties': {'is_mammal': {'description': 'Boolean value answering whether or not the given animal is a mammal', 'title': 'Is Mammal', 'type': 'boolean'}}, 'required': ['is_mammal']}, 'Invalid json output: Here is the response in the requested JSON format:\n\n{"is_mammal": false}\n\nThis indicates that an automobile is not a mammal, as it does not meet the criteria of being a living organism with warm blood and hair or fur.', 'Invalid json output: Here\'s the answer in JSON format, as required:\n\n{"properties": {"is_mammal": {"description": "Boolean value answering whether or not the given animal is a mammal", "title": "Is Mammal", "type": "boolean"}}, "required": ["is_mammal"]}\n false']


### Counter prompting

The LLama3.1 model performed very well, however both of our other models did very poorly. We will try it again with coutner prompting and few shot.


```python
prompt_counter = """Please tell me if the following creature is a mammal: {}.
                    \n
                    Answer in JSON format with a single key of 'is_mammal' and a value of 
                    'true' if it is a mammal or 'false' if it is not a mammal.
                    \n
                    Do not include a preamble or explanation of the answer,
                    do not format the JSON output as markdown text,
                    only return properly formatted JSON."""
```


```python
results = test_prompt_template(prompt_counter, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 324
    	Failures: 676
    	Failure rate: 67.60%
    	Total time: 235.21 seconds
    	Average time per request: 0.23 seconds
    llama3.2
    	Successes: 989
    	Failures: 11
    	Failure rate: 1.10%
    	Total time: 227.21 seconds
    	Average time per request: 0.23 seconds
    llama3.1
    	Successes: 1000
    	Failures: 0
    	Failure rate: 0.00%
    	Total time: 268.71 seconds
    	Average time per request: 0.27 seconds


### Few shot


```python
prompt_fewshot = """Please tell me if the following creature is a mammal: {}.\n
                    Answer in JSON format with a single key of 'is_mammal' and a value of 
                    'true' if it is a mammal or 'false' if it is not a mammal.
                    \n
                    Do not include a preamble or explanation of the answer,
                    do not format the JSON output as markdown text,
                    only return properly formatted JSON.
                    \n\n
                    For example, when asked if an "ocelot" is a mammal, your response
                    should be:\n
                    {{"is_mammal": "true"}}
                    \n\n
                    When asked if a "velociraptor" is a mammal, your response should be:\n
                    {{"is_mammal": "false"}}
                    \n\n
                    When asked if a "mid size sedan" is a mammal, your response should be:\n
                    {{"is_mammal": "false"}}"""
```


```python
results = test_prompt_template(prompt_fewshot, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 408
    	Failures: 592
    	Failure rate: 59.20%
    	Total time: 216.38 seconds
    	Average time per request: 0.21 seconds
    llama3.2
    	Successes: 1000
    	Failures: 0
    	Failure rate: 0.00%
    	Total time: 235.69 seconds
    	Average time per request: 0.24 seconds
    llama3.1
    	Successes: 1000
    	Failures: 0
    	Failure rate: 0.00%
    	Total time: 296.79 seconds
    	Average time per request: 0.30 seconds



```python
print("Llama 3.1 response sample:")
print(results['llama3.1']['responses'][:5])
print("-"*80)
print("Llama 3.2:1b response sample:")
print(results['llama3.2:1b']['responses'][:5])
print("-"*80)
print("Llama 3.2:3b response sample:")
print(results['llama3.2']['responses'][:5])
```

    Llama 3.1 response sample:
    [{'is_mammal': 'false'}, {'is_mammal': 'true'}, {'is_mammal': 'false'}, {'is_mammal': 'false'}, {'is_mammal': 'true'}]
    --------------------------------------------------------------------------------
    Llama 3.2:1b response sample:
    [{'is_mammal': True}, '{"is_mammal": "true"}', None, "argument of type 'NoneType' is not iterable", {'is_mammal': True}]
    --------------------------------------------------------------------------------
    Llama 3.2:3b response sample:
    [{'is_mammal': 'false'}, {'is_mammal': 'false'}, {'is_mammal': False}, {'is_mammal': 'false'}, {'is_mammal': 'false'}]



```python
results['llama3.2:1b']['responses'][:10], results['llama3.2:1b']['responses'][18:23]
```




    ([{'is_mammal': True},
      '{"is_mammal": "true"}',
      None,
      "argument of type 'NoneType' is not iterable",
      {'is_mammal': True},
      {'is_mammal': True},
      {'is_mammal': True},
      {'is_mammal': True},
      {'properties': {'is_mammal': False}},
      {'properties': {'is_mammal': True}}],
     [{'properties': {'is_mammal': 'true'}},
      {'$ref': '#/definitions/mammal'},
      {'properties': {'is_mammal': False}},
      {'properties': {'is_mammal': False}},
      {'is_mammal': 'true'}])



`LLama 3.2:1b` is going off the rails quite a bit with it's responses. It is adding keys, giving explanations, sometimes including seemingly random information.


## 5. Pydantic + `with_structured_output()` method

We will enforce a schema based upon Pydantic models using the `with_structured_output()` LLM method


```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
```

Define supporting Pydantic classes


```python
class IsMammal(BaseModel):
    """Class indicating whether or not a given creature is a mammal as a boolean value"""
    is_mammal: bool = Field(
        default=None, description="Boolean indicating whether a given creature is a mammal"
    )
```


```python
def test_prompt_template(prompt, option_list=[], iterations=1000, models=["llama3.2:1b", "llama3.2", "llama3.1"]):
    # Initialize an empty dictionary to store the responses and statistics
    results = {}

    for model in models:
        print("running {} iterations for {}".format(iterations, model))
        start_time = time.time()
        
        # Initialize lists for storing successes, failures, and responses
        successes = 0
        failures = 0
        responses = []
        
        for i in range(iterations):
            chat = ChatOllama(model=model)

            creature_input = option_list[random.randint(0, len(option_list)-1)]

            # instantiate llm w/ structured output: IsMammal class
            chain = prompt | chat.with_structured_output(schema=IsMammal)
            
            try:
                rsp = chain.invoke({"text": creature_input})
                responses.append(rsp)  # Store the response in the list
            
                # verify the expected data structure (IsMammal instance) was returned
                if rsp.is_mammal or not rsp.is_mammal:
                    successes += 1
            except Exception as e:
                responses.append(str(e))
                failures += 1  # Store the iteration number if failed
        
        stop_time = time.time()

        # Store the results in the dictionary using the model name as the key
        results[model] = {
            "successes": successes,
            "failures": failures,
            "responses": responses,
            "total_time": stop_time - start_time,
            "failure_rate": (failures / iterations) * 100.0
        }

    print("\nResults")
    print("-"*40)
    # Optionally, print the results if you want to see the result
    for model, data in results.items():
        print(model)
        print("\tSuccesses:", data['successes'])
        print("\tFailures:", data['failures'])
        print(f"\tFailure rate: {data['failure_rate']:.2f}%")
        print(f"\tTotal time: {data['total_time']:.2f} seconds")

        print(f"\tAverage time per request: {data['total_time']/len(data['responses']):.2f} seconds")

    return results
```

In an effort to determine how well the core structured output functionality works without any extra help, we will provide no directions in terms of output formatting, instead leaving it entirely up to the chat model to extract the appropriate information and return it in the expected class instance. So the input will be nothing but the given 'creature' to evaluate with no further explanation or instruction.


```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)
```


```python
results = test_prompt_template(prompt, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 916
    	Failures: 84
    	Failure rate: 8.40%
    	Total time: 275.16 seconds
    	Average time per request: 0.27 seconds
    llama3.2
    	Successes: 965
    	Failures: 35
    	Failure rate: 3.50%
    	Total time: 265.72 seconds
    	Average time per request: 0.26 seconds
    llama3.1
    	Successes: 876
    	Failures: 124
    	Failure rate: 12.40%
    	Total time: 442.71 seconds
    	Average time per request: 0.40 seconds


These results are rather disappointing. Now we will try some basic formatting instructions.


```python
prompt_help = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value. Return the value in"
            "the prescribed boolean format as an instance of the provided class.",
        ),
        ("human", """Is the following creature a mammal: {text}
                    \n\n
                    Only respond in the specified class structure format with a field
                    that is a boolean value."""),
    ]
)
```


```python
results = test_prompt_template(prompt_help, option_list=creatures, iterations=iterations_per_eval)
```

    running 1000 iterations for llama3.2:1b
    running 1000 iterations for llama3.2
    running 1000 iterations for llama3.1
    
    Results
    ----------------------------------------
    llama3.2:1b
    	Successes: 999
    	Failures: 1
    	Failure rate: 0.10%
    	Total time: 246.89 seconds
    	Average time per request: 0.25 seconds
    llama3.2
    	Successes: 999
    	Failures: 1
    	Failure rate: 0.10%
    	Total time: 282.65 seconds
    	Average time per request: 0.28 seconds
    llama3.1
    	Successes: 877
    	Failures: 123
    	Failure rate: 12.30%
    	Total time: 397.34 seconds
    	Average time per request: 0.35 seconds


Interestingly the newer, but smaller and presumably less capable models perform significantly better than the larger model in this case. This is very likely due to the given models instruction tuning and the formatting instructions being passed into the model.

# Conclusions

__REWRITE__
-----------

It's interesting that the langchain JSON formatting instructions seem to be derailing the model performance so much. The worst performer, `llama 3.2:1b`, is the easiest to illustrate this point. The few shot prompting produces an error rate of 2.70% but the few shot plus langchain JSON parser formatting instructions produces an error rate of 59.20%. These instructions contain their own few shot examples but they are more generic, explaining the JSON format itself, which in some cases seem to be leading the model toward this fictitious schema. While the langchain JSON parsing can be very helpful for outputting dictionaries for easy local code reference, the formatting instructions do not appear to produce more reliable output in smaller models.

The only possible caveat here is that the success metric for the JSON output parser with formatting instructions depended upon valid JSON as well as the presence of a root level key `is_mammal`. This final step of requiring the root key was not required of the prompt-only methods, but malformed JSON schema was not seen in spot checking generated output. I believe the few shot examples included in the langchain formatting instructions, while generic and intended for general purpose use, is confusing the models into producing undesirable schema. The same methodology applied in a domain specific form, as we did in the [Few shot](##-3-few-shot) examples, performs exceptionally well.

The langchain JSON parser formatting instructions are as follows:


```python
print(parser.get_format_instructions())
```

    The output should be formatted as a JSON instance that conforms to the JSON schema below.
    
    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
    
    Here is the output schema:
    ```
    {"properties": {"is_mammal": {"description": "Boolean value answering whether or not the given animal is a mammal", "title": "Is Mammal", "type": "boolean"}}, "required": ["is_mammal"]}
    ```



```python

```
