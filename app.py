import cohere
import os
import chainlit as cl
from dotenv import load_dotenv
from typing import List, Optional

from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

cohere_client = cohere.Client(cohere_api_key)


class CohereSummarize(LLM):
    model: str = "command"
    length: str = "short"
    extractiveness: str = "high"
    format: str = "bullets"

    @property
    def _llm_type(self) -> str:
        return "Cohere Summarize Model"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("Stop Kwargs are not permitted")
        if len(prompt) <= 250:
            return prompt

        response = cohere_client.summarize(
            text=prompt,
            model=self.model,
            length=self.length,
            extractiveness=self.extractiveness,
            format=self.format,
        )

        return response.summary


@cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Building Index...")

    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)

    fs = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, fs, namespace=embeddings.model
    )

    vector_store = FAISS.load_local("faiss_index", embeddings=cached_embedder)
    document_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    compressor = CohereRerank()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=document_retriever
    )

    msg.content = f"Index built!"
    await msg.send()

    msg.content = f"Setting up Chains!"
    await msg.send()

    llm = Cohere(cohere_api_key=cohere_api_key, temperature=0, frequency_penalty=0.75)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

    summary_llm = CohereSummarize()

    SUMMARY_PROMPT_TEMPLATE = """
    {context_to_summarize}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["context_to_summarize"], template=SUMMARY_PROMPT_TEMPLATE
    )

    summary_chain = LLMChain(llm=summary_llm, prompt=summary_prompt_template)

    combined_chain = SimpleSequentialChain(chains=[chain, summary_chain], verbose=True)

    msg.content = f"Setup Complete!"
    await msg.send()

    cl.user_session.set("chain", combined_chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    res = chain(message)
    await cl.Message(content=res["output"]).send()
