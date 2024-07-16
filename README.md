# Summarization

This service uses [langchain](https://python.langchain.com/docs/get_started/introduction) and LLMs to create summaries of documents.
The technique used for the summarization is the [Map/reduce](https://python.langchain.com/v0.2/docs/tutorials/summarization/#map-reduce) technique showed by langchain.
Each document will be summarized individually (Mapping phase) and
then all the summaries will be combined into a single summary (Reduce phase).
