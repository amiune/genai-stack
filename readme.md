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