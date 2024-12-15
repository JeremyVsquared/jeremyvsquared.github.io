---
title: FastAPI basics
date: 2024-12-15
---

# FastAPI basics

First we will create a local directory to save files to.


```python
!pwd
```

    /Users/jeremyvanvalkenburg/Repositories/posts



```python
import requests, json

import os
if not os.path.isdir('fastapi-basics'):
    os.mkdir('fastapi-basics')
```

## 1. Parameters with validation

This will write the `fastapi-basics/api.py` file for editing in file and running the dev server.


```python
%%writefile fastapi-basics/api.py

from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


class ModelBuild(BaseModel):
    id: int | None = None
    name: ModelName
    description: str
    version: float | None = None


app = FastAPI()


@app.get("/models/list")
async def list_models(skip: int=0, limit: int=10):
    """
    Returns a list of available models. 

    Adding optional GET parameters not in URL definition
    - ex: /models/list?skip=1&limit=5
    - ex: /models/list?skip=1

    This documentation will be added to the generated API docs.

    Parameters:
    - **skip**: (optional) how many initial items to skip in return value; default 0
    - **limit**: (optional) how many items to return; default 10
    """
    vals = [m.value for m in ModelName]

    return {
        "success": 1,
        "models": vals[skip:skip + limit]
    }


@app.get('/models/{model_name}')
async def get_model(model_name: ModelName):
    # predefined GET parameters
    # ex: /models/resnet
    if model_name is ModelName.alexnet:
        return {
            "success": 1,
            "model_name": model_name,
            "message": "deep learning ftw"
        }
    elif model_name is ModelName.lenet:
        return {
            "success": 1,
            "model_name": model_name,
            "message": "lecnn all the images"
        }
    elif model_name is ModelName.resnet:
        return {
            "success": 1,
            "model_name": model_name,
            "message": "have some residuals"
        }
    else:
        return {
            "success": 0,
            "model_name": model_name,
            "message": "unknown model"
        }


@app.post('/builds/create')
async def create_model(model: ModelBuild):
    # pretending to store the model...

    return {
        "success": 1,
        "message": "model created",
        "model": model
    }


@app.put('/builds/{model_id}')
async def update_model(model_id: int, model: ModelBuild):
    # pretending to update the model...

    return {
        "success": 1, 
        "message": "model updated",
        "model": model
    }
```

    Overwriting fastapi-basics/api.py


Run the API with the command

```
fastapi dev fastapi-basics/api.py
```

The server automatically update as the files are overwritten so stopping and restarting the server will not be necessary so long as you are editing or overwriting the specified file. 

You can interact with it through local requests or the following `requests` calls.


```python
rsp = requests.get(
    "http://127.0.0.1:8000/models/list"
)

json.loads(rsp.content)
```




    {'success': 1, 'models': ['alexnet', 'resnet', 'lenet']}



Trying unsupported methods


```python
rsp = requests.get(
    "http://127.0.0.1:8000/builds/create"
)

json.loads(rsp.content)
```




    {'detail': 'Method Not Allowed'}



Trying requests with mulitple missing values required by the model


```python
d = {
    "version": 0.2
}

rsp = requests.post(
    "http://127.0.0.1:8000/builds/create",
    json=d
)

json.loads(rsp.content)
```




    {'detail': [{'type': 'missing',
       'loc': ['body', 'name'],
       'msg': 'Field required',
       'input': {'version': 0.2}},
      {'type': 'missing',
       'loc': ['body', 'description'],
       'msg': 'Field required',
       'input': {'version': 0.2}}]}



Trying paramaeters with the wrong data type


```python
d = {
    "name": "resnet",
    "description": "CNN",
    "version": "Not a number"
}

rsp = requests.post(
    "http://127.0.0.1:8000/builds/create",
    json=d
)

json.loads(rsp.content)
```




    {'detail': [{'type': 'float_parsing',
       'loc': ['body', 'version'],
       'msg': 'Input should be a valid number, unable to parse string as a number',
       'input': 'Not a number'}]}



Trying valid input and method


```python
d = {
    "name": "resnet",
    "description": "CNN",
    "version": 0.2
}

rsp = requests.post(
    "http://127.0.0.1:8000/builds/create",
    json=d
)

json.loads(rsp.content)
```




    {'success': 1,
     'message': 'model created',
     'model': {'id': None, 'name': 'resnet', 'description': 'CNN', 'version': 0.2}}



## 2. Multiple and mixed parameters

Here we demonstrate mixed path and request body parameters.


```python
%%writefile fastapi-basics/api.py

from pydantic import BaseModel
from fastapi import FastAPI


class User(BaseModel):
    id: int | None = None
    username: str


class ModelBuild(BaseModel):
    id: int | None = None
    name: str
    description: str
    version: float | None = None


app = FastAPI()


@app.put("/builds/{model_id}")
async def create_build(model_id: int, model: ModelBuild, user: User):
    print(f"Model ID: {model_id}")
    print(f"Model parameter: {model}")
    print(f"User parameter: {user}")

    return {
        "id": model_id,
        "model": model,
        "user": user
    }
```

    Overwriting fastapi-basics/api.py


Run the API with the command

```
fastapi dev fastapi-basics/api.py
```

if the server is not already running.


```python
d = {
    "user": {
        "username": "jerry"
    },
    "model": {
        "name": "resnet",
        "description": "cnn",
        "version": 0.2
    }
}

rsp = requests.put(
    "http://127.0.0.1:8000/builds/1", 
    json=d
)

json.loads(rsp.content)
```




    {'id': 1,
     'model': {'id': None, 'name': 'resnet', 'description': 'cnn', 'version': 0.2},
     'user': {'id': None, 'username': 'jerry'}}



## 3. Nested models

We can nest models within other models while still enjoying the built in translation and validation.


```python
%%writefile fastapi-basics/api.py

from pydantic import BaseModel
from fastapi import FastAPI


class User(BaseModel):
    id: int | None = None
    username: str


class ModelBuild(BaseModel):
    id: int | None = None
    name: str
    description: str
    version: float | None = None
    contributors: list[User] | None = None


app = FastAPI()


@app.put("/builds/{model_id}")
async def create_build(model_id: int, model: ModelBuild):
    print(f"Model ID: {model_id}")
    print(f"Model parameter: {model}")

    return {
        "id": model_id,
        "model": model
    }
```

    Overwriting fastapi-basics/api.py


The nested model will be validated as with any other field. Here the validation will require a valid list of translated `User` objects.


```python
d = {
    "name": "resnet",
    "description": "cnn",
    "version": 0.2,
    "contributors": {
        "username": "jerry"
    }
}

rsp = requests.put(
    "http://127.0.0.1:8000/builds/1", 
    json=d
)

json.loads(rsp.content)
```




    {'detail': [{'type': 'list_type',
       'loc': ['body', 'contributors'],
       'msg': 'Input should be a valid list',
       'input': {'username': 'jerry'}}]}



Trying valid input


```python
d = {
    "name": "resnet",
    "description": "cnn",
    "version": 0.2,
    "contributors": [{
        "username": "jerry"
    },{
        "username": "jane"
    }]
}

rsp = requests.put(
    "http://127.0.0.1:8000/builds/1", 
    json=d
)

json.loads(rsp.content)
```




    {'id': 1,
     'model': {'id': None,
      'name': 'resnet',
      'description': 'cnn',
      'version': 0.2,
      'contributors': [{'id': None, 'username': 'jerry'},
       {'id': None, 'username': 'jane'}]}}



## 4. Splitting the application across mulitple files


```python
%%writefile fastapi-basics/api.py

from fastapi import FastAPI
import users, models


app = FastAPI()

app.include_router(users.router)
app.include_router(models.router)


@app.get("/")
async def home():
    return {
        "success": 1,
        "message": "hey there from the home page"
    }
```

    Overwriting fastapi-basics/api.py



```python
%%writefile fastapi-basics/users.py

from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter()


class User(BaseModel):
    id: int | None = None
    username: str


@router.get("/users/list")
async def list_users():
    # pretend to query from database
    users = [
        User(**{"id": 1, "username": "jerry"}),
        User(**{"id": 2, "username": "jane"})
    ]

    return users
```

    Overwriting fastapi-basics/users.py



```python
%%writefile fastapi-basics/models.py

from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter()


class Model(BaseModel):
    id: int | None = None
    name: str
    description: str
    version: float | None = None


@router.get("/models/list")
async def list_models():
    # pretend to query from database
    models = [
        Model(**{
            "id": 1,
            "name": "resnet",
            "description": "CNN",
            "version": 1.0
        }),
        Model(**{
            "id": 2,
            "name": "resnet",
            "description": "CNN",
            "version": 2.0
        }),
        Model(**{
            "id": 3,
            "name": "efficientnet",
            "description": "CNN"
        })
    ]

    return models
```

    Overwriting fastapi-basics/models.py



```python
rsp_users = requests.get("http://127.0.0.1:8000/users/list")
print(f"/users/list: {rsp_users.content}")
print()
rsp_models = requests.get("http://127.0.0.1:8000/models/list")
print(f"/models/list: {rsp_models.content}")
```

    /users/list: b'[{"id":1,"username":"jerry"},{"id":2,"username":"jane"}]'
    
    /models/list: b'[{"id":1,"name":"resnet","description":"CNN","version":1.0},{"id":2,"name":"resnet","description":"CNN","version":2.0},{"id":3,"name":"efficientnet","description":"CNN","version":null}]'


It is very likely that every path defined within `users.py` will be prefixed with `/users`. The same is true for `models.py` and `/models`. We can simplify our isolated library files a bit by adding a `prefix` parameter to the the `router`. This will allow us to reduce the path complexity in each file while avoiding potential path name collisions. The same functionality supports `tags` and `dependencies`.


```python
%%writefile fastapi-basics/users.py

from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter(
    prefix='/users',
    tags=['users']
)


class User(BaseModel):
    id: int | None = None
    username: str


@router.get("/list")
async def list_users():
    # pretend to query from database
    users = [
        User(**{"id": 1, "username": "jerry"}),
        User(**{"id": 2, "username": "jane"})
    ]

    return users
```

    Overwriting fastapi-basics/users.py



```python
%%writefile fastapi-basics/models.py

from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter(
    prefix='/models',
    tags=['models']
)


class Model(BaseModel):
    id: int | None = None
    name: str
    description: str
    version: float | None = None


@router.get("/list")
async def list_models():
    # pretend to query from database
    models = [
        Model(**{
            "id": 1,
            "name": "resnet",
            "description": "CNN",
            "version": 1.0
        }),
        Model(**{
            "id": 2,
            "name": "resnet",
            "description": "CNN",
            "version": 2.0
        }),
        Model(**{
            "id": 3,
            "name": "efficientnet",
            "description": "CNN"
        })
    ]

    return models
```

    Overwriting fastapi-basics/models.py



```python
rsp_users = requests.get("http://127.0.0.1:8000/users/list")
print(f"/users/list: {rsp_users.content}")
print()
rsp_models = requests.get("http://127.0.0.1:8000/models/list")
print(f"/models/list: {rsp_models.content}")
```

    /users/list: b'[{"id":1,"username":"jerry"},{"id":2,"username":"jane"}]'
    
    /models/list: b'[{"id":1,"name":"resnet","description":"CNN","version":1.0},{"id":2,"name":"resnet","description":"CNN","version":2.0},{"id":3,"name":"efficientnet","description":"CNN","version":null}]'


## 5. Sockets & testing


```python
%%writefile fastapi-basics/api.py

from datetime import datetime
from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from fastapi.testclient import TestClient
from pydantic import BaseModel


app = FastAPI()


class Metric(BaseModel):
    timestep: datetime | int
    metric_type: str
    value: float


class Model(BaseModel):
    id: int | None = None
    name: str
    description: str | None = None
    version: float | None = None
    metrics: list[Metric] | None = None


@app.websocket("/model/{model_id}/metrics/add")
async def add_metric_for_model(socket: WebSocket, model_id: int):
    # pretend to load model data from database
    model = Model(
        id=model_id,
        name="darnn",
        version=0.8,
        metrics=[]
    )

    await socket.accept()
    try:
        while True:
            data = await socket.receive_json()
            try:
                print(f"DEBUG: received data: {data}")

                metric = Metric(**data)
                model.metrics.append(metric)
                response = {
                    "success": 1,
                    "metric": metric.model_dump_json()
                }
            except Exception as e:
                response = {
                    'success': 0, 
                    'error': str(e)
                }
            await socket.send_json(response)
    except (WebSocketDisconnect, ConnectionClosed):
        print("DEBUG: client disconnected")


def test_websocket():
    client = TestClient(app)
    test_url = "/model/1/metrics/add"
    with client.websocket_connect(test_url) as socket:
        print("DEBUG: connected to socket")

        socket.send_json({
            "timestep": 1,
            "metric_type": "R2",
            "value": 0.2312
        })
        data = socket.receive_json()
        
        print(f"DEBUG: data received: {data}")

        assert 'success' in data
        print(f"DEBUG: testing output: {data['success']}")
        assert data['success']==1

        socket
```

    Overwriting fastapi-basics/api.py



```python
!pytest fastapi-basics/api.py
```

    /Users/jeremyvanvalkenburg/anaconda3/lib/python3.11/site-packages/pytest_asyncio/plugin.py:207: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
    The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"
    
      warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
    [1m============================= test session starts ==============================[0m
    platform darwin -- Python 3.11.8, pytest-8.3.4, pluggy-1.5.0
    rootdir: /Users/jeremyvanvalkenburg/Repositories/posts
    plugins: asyncio-0.25.2, anyio-4.6.0, dash-2.14.2
    asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None
    collected 1 item                                                               [0m
    
    fastapi-basics/api.py [32m.[0m[32m                                                  [100%][0m
    
    [32m============================== [32m[1m1 passed[0m[32m in 0.32s[0m[32m ===============================[0m



```python

```

# Clean up

Delete the generated directory.


```python
import shutil

if os.path.isdir('fastapi-basics'):
    shutil.rmtree('fastapi-basics')
    print("deleted directory")
else:
    print("could not locate directory to delete")
```

    deleted directory



```python

```


```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
```


```python
pt1 = Point(32, 64)
pt1
```




    Point(x=32, y=64)




```python

```
