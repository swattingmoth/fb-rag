# Facebook Rag Pipeline

## Initial Setup
- Install [Docker](https://www.docker.com/)
- Use Docker to [pull and run Qdrant](https://qdrant.tech/documentation/guides/installation/#docker
)
- Export your [Facebook Data](https://www.facebook.com/help/212802592074644/?helpref=uf_share) and unzip to a local folder
- Install [Ollama](https://ollama.com/download)
- Pull `qwen3-embedding:8b` and `qwen3:14b` using ollama. If you have limited RAM and/or GPU memory you can try `qwen3:8b` instead of `qwen3:14b`.
```
ollama pull qwen3-embedding:8b
ollama pull qwen3:14b
```

## Local repo set up
- Clone this git repo
- Copy `sample_config.json` to `config.json`. Replace settings as necessary.
  - `collection_name` the name of the qdrant collection to create
  - `document_path` the json file in which to store the cleaned and processed Facebook data
  -  `log_folder` the folder to write the log files to
  -  `raw_data_path` The path to the raw facebook post data
  -  `model` The LLM model to use to generate answers. 
  -  `qdrant_host` Where qdrant is hosted
  - `qdrant_port` The port where qrdant is hosted
- Install the required packages: 
`pip install -r requirements.txt`

# Database set up
- Run `facebook_processor.py`. This will process the Facebook posts stored at `raw_data_path` and store them at `document_path`. **Note** I don't use Facebook a lot so I have less than 300 total posts. If you have more posts, modifications may be required.
- Make sure Qdrant is running and run `store_docs.py`. 


