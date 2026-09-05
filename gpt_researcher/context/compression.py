"""Context compression utilities for GPT Researcher.

This module provides classes for compressing and retrieving relevant
context from documents using embeddings and similarity filtering.

The compression pipeline:
1. Splits documents into chunks
2. Filters chunks by embedding similarity to the query
3. Returns the most relevant chunks as context

Classes:
    VectorstoreCompressor: Retrieves context from a vector store.
    ContextCompressor: Compresses raw documents using embedding similarity.
    WrittenContentCompressor: Compresses previously written content sections.
"""

import asyncio
import hashlib
import logging
import os
from typing import Optional

import numpy as np
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost
from ..vector_store import VectorStoreWrapper
from .retriever import SearchAPIRetriever, SectionRetriever

_compression_logger = logging.getLogger(__name__)


class _EmbeddingCache:
    """Thread-safe cache for embedding vectors computed during compression."""

    def __init__(self):
        self.doc_embeddings: dict[str, list[float]] = {}
        self.query_embeddings: dict[str, list[float]] = {}
        self.last_kept_ids: list[str] = []
        self.last_pruned_ids: list[str] = []
        self.last_similarities: dict[str, float] = {}

    @staticmethod
    def _chunk_id(text: str, source: str = "") -> str:
        raw = f"{source}:{text[:200]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def record(
        self,
        all_chunks: list[Document],
        kept_chunks: list[Document],
        chunk_embeddings: list[list[float]],
        query_embedding: list[float],
        query: str,
        similarity_threshold: float,
    ):
        self.query_embeddings[query] = query_embedding

        kept_contents = {d.page_content for d in kept_chunks}
        kept_ids = []
        pruned_ids = []

        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)

        for chunk, emb in zip(all_chunks, chunk_embeddings):
            cid = self._chunk_id(chunk.page_content, chunk.metadata.get("source", ""))
            self.doc_embeddings[cid] = emb

            d_vec = np.array(emb)
            d_norm = np.linalg.norm(d_vec)
            sim = float(np.dot(q_vec, d_vec) / (q_norm * d_norm)) if q_norm and d_norm else 0.0
            self.last_similarities[cid] = sim

            if chunk.page_content in kept_contents:
                kept_ids.append(cid)
            else:
                pruned_ids.append(cid)

        self.last_kept_ids = kept_ids
        self.last_pruned_ids = pruned_ids

        _compression_logger.info(
            f"Embedding cache: {len(kept_ids)} kept, {len(pruned_ids)} pruned "
            f"(threshold={similarity_threshold})"
        )


_global_embedding_cache = _EmbeddingCache()


class VectorstoreCompressor:
    """Retrieves and compresses context from a vector store.

    Uses similarity search on an existing vector store to find
    relevant documents for a given query.

    Attributes:
        vector_store: The vector store wrapper to search.
        max_results: Maximum number of results to return.
        filter: Optional filter for vector store queries.
    """

    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: Optional[dict] = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the VectorstoreCompressor.

        Args:
            vector_store: The vector store to search.
            max_results: Maximum number of results to return.
            filter: Optional filter dictionary for queries.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter
        self.kwargs = kwargs
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from the vector store.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted string of relevant document content.
        """
        results = await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)


class ContextCompressor:
    """Compresses raw documents to extract relevant context.

    Uses embedding similarity to filter document chunks and return
    only the most relevant content for a given query.

    Attributes:
        documents: List of documents to compress.
        embeddings: Embedding model for similarity calculation.
        max_results: Maximum number of results to return.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        similarity_threshold: float | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the ContextCompressor.

        Args:
            documents: List of documents to compress.
            embeddings: Embedding model instance.
            max_results: Maximum number of results to return.
            similarity_threshold: Minimum similarity score for inclusion.
                Falls back to the SIMILARITY_THRESHOLD env var when not given.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        if similarity_threshold is None:
            similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))
        self.similarity_threshold = similarity_threshold
        self.prompt_family = prompt_family

    def __get_contextual_retriever(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(
            pages=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', '') or '',
                    metadata={
                        "title": doc.get("title", "") or "",
                        "source": doc.get("source") or doc.get("url") or "",
                    },
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        base_retriever = SearchAPIRetriever(pages=self.documents)
        raw_docs = base_retriever.invoke(query, **self.kwargs)
        all_chunks = splitter.transform_documents(raw_docs)

        if not all_chunks:
            return ""

        chunk_texts = [c.page_content for c in all_chunks]
        chunk_embeddings = await asyncio.to_thread(self.embeddings.embed_documents, chunk_texts)
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)

        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        kept_chunks = []
        for chunk, emb in zip(all_chunks, chunk_embeddings):
            d_vec = np.array(emb)
            d_norm = np.linalg.norm(d_vec)
            sim = float(np.dot(q_vec, d_vec) / (q_norm * d_norm)) if q_norm and d_norm else 0.0
            if sim >= self.similarity_threshold:
                kept_chunks.append(chunk)

        _global_embedding_cache.record(
            all_chunks=all_chunks,
            kept_chunks=kept_chunks,
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_embedding,
            query=query,
            similarity_threshold=self.similarity_threshold,
        )

        return self.prompt_family.pretty_print_docs(kept_chunks, max_results)


class WrittenContentCompressor:
    """Compresses previously written content sections.

    Specialized compressor for finding relevant sections from
    previously written report content, preserving section titles
    and structure.

    Attributes:
        documents: List of written content sections.
        embeddings: Embedding model for similarity calculation.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(self, documents, embeddings, similarity_threshold: float, **kwargs):
        """Initialize the WrittenContentCompressor.

        Args:
            documents: List of written content sections.
            embeddings: Embedding model instance.
            similarity_threshold: Minimum similarity score for inclusion.
            **kwargs: Additional keyword arguments.
        """
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def __get_contextual_retriever(self):
        """Build the contextual compression retriever for sections.

        Returns:
            A ContextualCompressionRetriever configured for section retrieval.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SectionRetriever(
            sections=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    def __pretty_docs_list(self, docs, top_n: int) -> list[str]:
        """Format documents as a list of title/content strings.

        Args:
            docs: List of documents to format.
            top_n: Maximum number of documents to include.

        Returns:
            List of formatted document strings.
        """
        return [f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n" for i, d in enumerate(docs) if i < top_n]

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> list[str]:
        """Get relevant written content sections asynchronously.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            List of formatted section strings.
        """
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.__pretty_docs_list(relevant_docs, max_results)
