---
layout: post
title: Development in Jupyter Notebooks
date: 2024-07-01
description: Jupyter features useful for general development in the notebook environment
---

# Developing in Jupyter notebooks

## Writing to file

The keyword 

`%%writefile filename.py` 

at the beginning of a cell can be used to write (create or overwrite) the contents of the cell to a file. The suffix `-a` will append to an existing file. 

`%%writefile -a filename.py`

## `nbconvert`

The contents of a notebook can be exported to a nmarkdown file using the terminal command

`jupyter nbconvert --to markdown notebook_file.ipynb`

`nbconvert` can target multiple files simultaneously. This demonstration will convert to HTML.

`jupyter nbconvert --to html notebooks/*.ipynb`

By default, the entirety of the notebook will be exported. If only the outputs are wanted (omitting the code blocks) the parameter `--no-input` can be added to only export the cell outputs such as print output and visualizations.

`jupyter nbconvert --to markdown --no-input notebook_file.ipynb`

Additionally, specific cells can be omitted from export by editing the cell metadata with the following:

```json
{
    "nbconvert": {
        "exclude": true
    }
}
```

or by adding the `hide_input` tag. This is particularly useful to omit unit tests and export directly to library files.

It is possible to execute all cells in a notebook using `nbconvert` as well. This can be useful to verify functionality or update outputs prior to export. This can be done with the `--execute` parameter. By default this will save a newly executed version to a new file but can be saved to the original file with the `--inplace` parameter.

`jupyter nbconvert --to markdown --no-input --execute --inplace notebook_file.ipynb`

The output file can be specified using the `--output` parameter. This is especially useful when developing scripts in a notebook and exporting to a script.

`jupyter nbconvert --to python lib_dev.ipynb --output ../lib/lib.py`

In the case of developing python scripts in a notebook, it is often necessary to omit cell outputs as well.

`jupyter nbconvert --to python lib_dev.ipynb --output ../lib/lib.py --TemplateExporter.exclude_output=True`

## Performance

The keyword `%time` can be used to time cell execution if the interface doesn't show this by default. Even in such cases, it can be an easy and consistent way to add performance metrics to output.


```python
%time
for i in range(10000):
    i**2
```

    CPU times: user 4 µs, sys: 0 ns, total: 4 µs
    Wall time: 7.15 µs


`%%timeit` can be used to benchmark code execution.


```python
%%timeit
for i in range(10000):
    i**2
```

    1.69 ms ± 12.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


## Display

Displaying varibles in the default presentation is sometimes desireble when it is not convenient or clean to end a cell with the given variable. For example, a `pandas.DataFrame` will be presented in different ways by the default rendering system and the `print()` function.


```python
import seaborn as sns
```


```python
df = sns.load_dataset('iris')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.head())
```

       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa


The default rendering as shown in the first output of `df.head()` can be called as a function from the `IPython.display` library.


```python
from IPython.display import display
```


```python
display(df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>


This function will support the default notebook presentation of a variable when not present as the last line of the cell execution.

## Variable inspection


```python
%whos
```

    Variable   Type    Data/Info
    ----------------------------
    i          int     9999


## Terminal commands

Terminal commands can be run from cells with the `!` prefix operator. For instance,

```
!pip list | grep torch
```

will perform this terminal command to list `pip` packages containing the string value _torch_


