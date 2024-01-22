# GenAI Stack
The GenAI Stack will get you started building your own GenAI application in no time.

# Configure

Available variables:
| Variable Name          | Default value                      | Description                                                             |
|------------------------|------------------------------------|-------------------------------------------------------------------------|
| OLLAMA_BASE_URL        | http://host.docker.internal:11434  | REQUIRED - URL to Ollama LLM API                                        |   
| WEAVIATE_URI           | weaviate://database:7687           | REQUIRED - URL to weaviate database                                     |
| WEAVIATE_USERNAME      | weaviate                           | REQUIRED - Username for weaviate database                               |
| WEAVIATE_PASSWORD      | password                           | REQUIRED - Password for weaviate database                               |
| LLM                    | llama2                             | REQUIRED - Can be any Ollama model tag                                  |
| EMBEDDING_MODEL        | sentence_transformer               | REQUIRED - Can be sentence_transformer or ollama                        |

## LLM Configuration
MacOS and Linux users can use any LLM that's available via Ollama. Check the "tags" section under the model page you want to use on https://ollama.ai/library and write the tag for the value of the environment variable `LLM=` in the `.env` file.
All platforms can use GPT-3.5-turbo and GPT-4 (bring your own API keys for OpenAI models).

**MacOS**
Install [Ollama](https://ollama.ai) on MacOS and start it before running `docker compose up`.

**Linux**
No need to install Ollama manually, it will run in a container as
part of the stack when running with the Linux profile: run `docker compose --profile linux up`.
Make sure to set the `OLLAMA_BASE_URL=http://llm:11434` in the `.env` file when using Ollama docker container.

To use the Linux-GPU profile: run `docker compose --profile linux-gpu up`. Also change `OLLAMA_BASE_URL=http://llm-gpu:11434` in the `.env` file.

### Verify that the pgvector plugin has been installed:
```
# connect to the local PG 
psql -U admin -d searchable -h 127.0.0.1
```

#### Check that the `searchable` DB is there
``` 
# check the DB list:
searchable-# \l
                              List of databases
    Name    | Owner | Encoding |  Collate   |   Ctype    | Access privileges 
------------+-------+----------+------------+------------+-------------------
 postgres   | admin | UTF8     | en_US.utf8 | en_US.utf8 | 
 searchable | admin | UTF8     | en_US.utf8 | en_US.utf8 | 
 template0  | admin | UTF8     | en_US.utf8 | en_US.utf8 | =c/admin         +
            |       |          |            |            | admin=CTc/admin
 template1  | admin | UTF8     | en_US.utf8 | en_US.utf8 | =c/admin         +
            |       |          |            |            | admin=CTc/admin
```

#### Making sure you're connected to the right DB
```
searchable-# \c searchable
```
#### check that the VECTOR plugin is installed
This can be done either by:
 `searchable-# select * from pg_extension;`
or via the psql command: 
 `searchable=# \dx;`

You should see something like this:

```
                             List of installed extensions
  Name   | Version |   Schema   |                     Description                      
---------+---------+------------+------------------------------------------------------
 plpgsql | 1.0     | pg_catalog | PL/pgSQL procedural language
 vector  | 0.5.1   | public     | vector data type and ivfflat and hnsw access methods
(2 rows)
```

If `VECTOR` is not present in the list then migrations are not being run correctly.

FYI you can install it doing the following:

```
searchable=# CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION
```


**Windows**
Not supported by Ollama, so Windows users need to generate an OpenAI API key and configure the stack to use `gpt-3.5` or `gpt-4` in the `.env` file.
# Develop

> [!WARNING]
> There is a performance issue that impacts python applications in the `4.24.x` releases of Docker Desktop. Please upgrade to the latest release before using this stack.

**To start everything**
```
docker compose up
```
If changes to build scripts have been made, **rebuild**.
```
docker compose up --build
```

To enter **watch mode** (auto rebuild on file changes).
First start everything, then in new terminal:
```
docker compose watch
```

**Shutdown**
If health check fails or containers don't start up as expected, shutdown
completely to start up again.
```
docker compose down
```

# Applications

| Name | Main files | Compose name | URLs | Description |
|---|---|---|---|---|
| Support Bot | `bot.py` | `bot` | http://localhost:8501 | Main usecase. Fullstack Python application. |

The database can be explored at http://localhost:7474.

## App - Agent Bot

UI: http://localhost:8501
DB client: http://localhost:7474

- answer support question based on recent entries
- provide summarized answers with sources

---