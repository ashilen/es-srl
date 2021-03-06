###EN.601.769 Assignment 1: Semantic Role Labeling

The repo is organized as follows:

[data](data) contains the splits for each prototype with their labeled predicate/argument pairs. It also contains the serialized models.
[srl](srl) contains the implementations for querying UDS and for building the models.

To start from scratch, install the requirements:
```bash
pip install -r requirements.txt
```

Then all of the models can be tested with:
```bash
python -m srl.model
```

To generate new data from UDS for a particular protorole:
```bash
python -m srl.query agent
```

And to build a new model and test it, first remove the appropriate serialization directory, and then:
```bash
python -m srl.model --protorole=agent
```


####python -m srl.query
```bash
usage: query.py [-h] [--split {dev,test,train}] [--limit LIMIT] [--stats] [--pretty] [--raw] [--uds_version {1.0,2.0}]
                {agent,patient,theme,experiencer,destination}

Query the UDS.

positional arguments:
  {agent,patient,theme,experiencer,destination}
                        Protorole to query for.

optional arguments:
  -h, --help            show this help message and exit
  --split {dev,test,train}
                        Data split to query on.
  --limit LIMIT
  --stats               Show some stats about the query.
  --pretty              Pretty print the query results.
  --raw                 Return the raw SPARQL responses.
  --uds_version {1.0,2.0}
  ```

####python -m srl.model
  ```bash
usage: model.py [-h] [--protorole {agent,patient,theme,experiencer,destination}]

Model the UDS.

optional arguments:
  -h, --help            show this help message and exit
  --protorole {agent,patient,theme,experiencer,destination}
                        Protorole to model.
```
