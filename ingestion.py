import asyncio
import logging
import os
import ssl
from dataclasses import dataclass
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logging

logger = logging.getLogger("RAG_Pipeline")

# Silence noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["SSL_CERT_DIR"] = certifi.where()


@dataclass(frozen=True)
class PipelineConfig:
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    embedding_model: str = "qwen3-embedding:0.6b"
    target_url: str = "https://python.langchain.com"
    crawl_chunk_size: int = 10
    index_batch_size: int = 40  # Slightly smaller batches to ease Ollama load

    def validate(self):
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set.")
        if not self.pinecone_index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set.")


class DocumentIngestionPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.validate()

        self.embeddings = OllamaEmbeddings(model=self.config.embedding_model)
        self.tavily_extract = TavilyExtract()
        self.tavily_map = TavilyMap(max_depth=2, max_pages=1000, max_concurrency=20)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " "]
        )

    def _chunk_urls(self, urls: List[str]) -> List[List[str]]:
        size = self.config.crawl_chunk_size
        return [urls[i : i + size] for i in range(0, len(urls), size)]

    async def _extract_batch(self, urls: List[str], batch_num: int) -> List[Document]:
        try:
            logger.info(
                f"Extracting content from batch {batch_num} ({len(urls)} URLs)..."
            )
            crawl_results = await self.tavily_extract.ainvoke({"urls": urls})
            return [
                Document(
                    page_content=result["raw_content"],
                    metadata={"source": result["url"]},
                )
                for result in crawl_results.get("results", [])
                if result.get("raw_content")
            ]
        except Exception as e:
            logger.error(f"Failed to extract content from batch {batch_num}: {e}")
            return []

    async def extract_all_urls(self, urls: List[str]) -> List[Document]:
        url_chunks = self._chunk_urls(urls)
        tasks = [
            self._extract_batch(chunk, i + 1) for i, chunk in enumerate(url_chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_documents = []
        for result in results:
            if not isinstance(result, Exception) and result:
                all_documents.extend(result)
        return all_documents

    async def index_documents(self, documents: List[Document]):
        """
        Production-grade high-concurrency indexer.
        Isolates client connection lifecycles to guarantee no HTTP session dropouts.
        """
        batch_size = self.config.index_batch_size
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        logger.info(f"Starting resilient parallel indexing for {len(batches)} batches.")

        # Semaphore limits simultaneous heavy tasks (Ollama computation + Pinecone network I/O)
        semaphore = asyncio.Semaphore(3)

        async def process_and_upload_batch(batch: List[Document], batch_num: int):
            async with semaphore:
                try:
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]

                    logger.info(
                        f"[{batch_num}/{len(batches)}] Generating embeddings locally..."
                    )
                    embedded_vectors = await self.embeddings.aembed_documents(texts)

                    logger.info(
                        f"[{batch_num}/{len(batches)}] Opening fresh session & uploading to Pinecone..."
                    )
                    vector_store = PineconeVectorStore(
                        index_name=self.config.pinecone_index_name,
                        pinecone_api_key=self.config.pinecone_api_key,
                        embedding=self.embeddings,
                    )

                    await vector_store.aadd_texts(
                        texts=texts, embeddings=embedded_vectors, metadatas=metadatas
                    )
                    logger.info(f"Successfully finalized batch {batch_num}.")
                    return True
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    return False

        # Build execution tasks
        tasks = [
            process_and_upload_batch(batch, i + 1) for i, batch in enumerate(batches)
        ]

        # Fire parallel workers
        results = await asyncio.gather(*tasks)

        failed_count = sum(1 for r in results if not r)
        if failed_count > 0:
            logger.warning(f"Pipeline completed with {failed_count} failed batches.")
        else:
            logger.info("All document batches successfully indexed into Pinecone!")

    async def run(self):
        logger.info(
            f"=== Starting Ingestion Pipeline for target: {self.config.target_url} ==="
        )

        site_map = await self.tavily_map.ainvoke(self.config.target_url)
        urls_to_crawl = site_map.get("results", [])
        if not urls_to_crawl:
            logger.error("No target endpoints discovered.")
            return {"status": "failed"}

        logger.info(f"Discovered {len(urls_to_crawl)} target endpoints.")

        documents = await self.extract_all_urls(urls_to_crawl)
        logger.info(f"Total raw pages retrieved: {len(documents)}")

        chunks = await self.text_splitter.atransform_documents(documents)
        logger.info(f"Generated {len(chunks)} text chunks for vectors.")

        await self.index_documents(chunks)

        logger.info("=== Ingestion Pipeline Lifecycle Finalized ===")
        return {"status": "success"}


if __name__ == "__main__":
    pipeline_config = PipelineConfig()
    pipeline = DocumentIngestionPipeline(config=pipeline_config)
    asyncio.run(pipeline.run())
