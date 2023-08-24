# 🐉 DnD Compendium Assistant 🐉

A walkthrough on how to create the Index that powers this application can be found [here](https://colab.research.google.com/drive/1bG60AghMKHX73UneqDYrtSZZ4wiczWOS?usp=sharing)

### Problem Statement

Dungeons and Dragons, while a fantastically fun and engaging game, has a lot of little rules that can matter very much. Looking up these fiddly rules is often time-consuming, and sometimes manual, process that interupts the flow of play.

> NOTE: Looking up information in the books is a rewarding experience - but not when you're double checking the 'Blindness' rules for the 200th time just to be sure

### The Solution

A Cohere-powered LangChain application that lets us query the SRD in natural language and provides natural language responses. 

Augmenting our LLM with some outside D&D knowledge will let us avoid the hassels of manually looking the information up, and spend more time playing the game!

The main components are: 

- [Cohere](https://docs.cohere.com/docs) - the LLM (and other services) that powers the application
- [LangChain](https://github.com/langchain-ai/langchain) - an LLM orchestration platform
