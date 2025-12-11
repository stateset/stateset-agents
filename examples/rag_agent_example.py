"""
RAG (Retrieval-Augmented Generation) Agent Example

This example demonstrates how to build a production-ready RAG agent using
StateSet Agents with vector database integration for knowledge retrieval.

Features:
- Document ingestion and chunking
- Vector embedding generation
- Semantic search with ChromaDB (or FAISS fallback)
- Context-aware response generation
- Source citation in responses

Requirements:
    pip install chromadb sentence-transformers faiss-cpu

Usage:
    python examples/rag_agent_example.py

    # Or with custom documents:
    python examples/rag_agent_example.py --docs /path/to/documents
"""

import argparse
import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Vector Store Abstraction
# ============================================================================


@dataclass
class Document:
    """Represents a document chunk for retrieval.

    Attributes:
        id: Unique document identifier
        content: Text content of the document
        metadata: Additional metadata (source, page, etc.)
        embedding: Optional pre-computed embedding
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def from_text(cls, text: str, source: str = "unknown") -> "Document":
        """Create a document from text with auto-generated ID."""
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return cls(id=doc_id, content=text, metadata={"source": source})


@dataclass
class RetrievalResult:
    """Result from a retrieval query.

    Attributes:
        document: The retrieved document
        score: Similarity score (higher = more similar)
        rank: Position in results (1 = best match)
    """

    document: Document
    score: float
    rank: int


class VectorStore:
    """Abstract vector store interface for RAG retrieval."""

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        raise NotImplementedError

    async def search(
        self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Search for similar documents."""
        raise NotImplementedError

    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID."""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence-transformers model for embeddings
            persist_directory: Directory to persist data (None for in-memory)
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "ChromaDB is required for RAG. Install with: pip install chromadb"
            )

        self.embedding_model = embedding_model

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Setup embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Initialized ChromaDB collection '{collection_name}' "
            f"with {self.collection.count()} documents"
        )

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB."""
        if not documents:
            return

        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    async def search(
        self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Search ChromaDB for similar documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        for i, (doc_id, content, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Convert distance to similarity score (cosine distance to similarity)
            score = 1 - distance

            retrieval_results.append(
                RetrievalResult(
                    document=Document(id=doc_id, content=content, metadata=metadata),
                    score=score,
                    rank=i + 1,
                )
            )

        return retrieval_results

    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        self.collection.delete(ids=document_ids)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store (fallback when ChromaDB unavailable)."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize FAISS vector store.

        Args:
            embedding_model: Sentence-transformers model for embeddings
        """
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "FAISS and sentence-transformers required. "
                "Install with: pip install faiss-cpu sentence-transformers"
            )

        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
        self.documents: Dict[int, Document] = {}
        self._next_id = 0

        logger.info(f"Initialized FAISS index with dimension {self.dimension}")

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS index."""
        import numpy as np

        if not documents:
            return

        # Generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self.model.encode(contents, normalize_embeddings=True)

        # Add to index
        self.index.add(np.array(embeddings).astype("float32"))

        # Store document mapping
        for doc in documents:
            self.documents[self._next_id] = doc
            self._next_id += 1

        logger.info(f"Added {len(documents)} documents to FAISS index")

    async def search(
        self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Search FAISS index for similar documents."""
        import numpy as np

        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Search
        scores, indices = self.index.search(
            np.array(query_embedding).astype("float32"), top_k
        )

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue

            doc = self.documents.get(idx)
            if doc:
                # Apply metadata filter if provided
                if filter_metadata:
                    if not all(
                        doc.metadata.get(k) == v for k, v in filter_metadata.items()
                    ):
                        continue

                results.append(
                    RetrievalResult(document=doc, score=float(score), rank=rank + 1)
                )

        return results

    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents (not efficiently supported in FAISS)."""
        logger.warning("FAISS deletion requires index rebuild - not implemented")


# ============================================================================
# Document Processing
# ============================================================================


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = "\n\n",
) -> List[str]:
    """Split text into overlapping chunks for better retrieval.

    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        separator: Preferred split point

    Returns:
        List of text chunks
    """
    # First split by separator
    sections = text.split(separator)

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section fits in current chunk, add it
        if len(current_chunk) + len(section) + len(separator) <= chunk_size:
            if current_chunk:
                current_chunk += separator
            current_chunk += section
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)

            # If section is larger than chunk_size, split it further
            if len(section) > chunk_size:
                # Split on sentences
                sentences = section.replace(". ", ".|").split("|")
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                current_chunk = section

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Add overlap by prepending end of previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
            overlapped_chunks.append(overlap_text + " " + chunks[i])
        chunks = overlapped_chunks

    return chunks


def load_documents_from_directory(
    directory: str,
    extensions: List[str] = [".txt", ".md", ".py"],
    chunk_size: int = 500,
) -> List[Document]:
    """Load and chunk documents from a directory.

    Args:
        directory: Path to directory containing documents
        extensions: File extensions to process
        chunk_size: Size of text chunks

    Returns:
        List of Document objects
    """
    documents = []
    directory = Path(directory)

    for ext in extensions:
        for filepath in directory.rglob(f"*{ext}"):
            try:
                content = filepath.read_text(encoding="utf-8")

                # Chunk the content
                chunks = chunk_text(content, chunk_size=chunk_size)

                # Create documents for each chunk
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        id=f"{filepath.stem}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}",
                        content=chunk,
                        metadata={
                            "source": str(filepath),
                            "filename": filepath.name,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "extension": ext,
                        },
                    )
                    documents.append(doc)

                logger.info(f"Loaded {len(chunks)} chunks from {filepath.name}")

            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

    return documents


# ============================================================================
# RAG Agent
# ============================================================================


class RAGAgent(MultiTurnAgent):
    """Agent with Retrieval-Augmented Generation capabilities.

    Combines vector-based document retrieval with LLM generation
    for knowledge-grounded responses.

    Example:
        >>> vector_store = ChromaVectorStore()
        >>> await vector_store.add_documents(documents)
        >>>
        >>> agent = RAGAgent(config, vector_store)
        >>> await agent.initialize()
        >>>
        >>> response = await agent.query("What is the return policy?")
        >>> print(response.answer)
        >>> print(response.sources)
    """

    def __init__(
        self,
        config: AgentConfig,
        vector_store: VectorStore,
        top_k: int = 5,
        min_relevance_score: float = 0.3,
        include_sources: bool = True,
        **kwargs,
    ):
        """Initialize RAG agent.

        Args:
            config: Agent configuration
            vector_store: Vector store for document retrieval
            top_k: Number of documents to retrieve
            min_relevance_score: Minimum score to include a document
            include_sources: Whether to include source citations
        """
        # Add RAG-specific system prompt
        rag_system_prompt = """You are a helpful AI assistant with access to a knowledge base.
When answering questions:
1. Use the provided context to give accurate, grounded answers
2. If the context doesn't contain relevant information, say so clearly
3. Cite your sources by mentioning which documents you're referencing
4. Don't make up information not present in the context
5. Be concise but thorough

Context will be provided in <context> tags before each question."""

        if config.system_prompt:
            config.system_prompt = rag_system_prompt + "\n\n" + config.system_prompt
        else:
            config.system_prompt = rag_system_prompt

        super().__init__(config, **kwargs)

        self.vector_store = vector_store
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.include_sources = include_sources

    async def retrieve(
        self, query: str, filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            filter_metadata: Optional metadata filter

        Returns:
            List of retrieval results sorted by relevance
        """
        results = await self.vector_store.search(
            query=query,
            top_k=self.top_k,
            filter_metadata=filter_metadata,
        )

        # Filter by minimum relevance score
        filtered = [r for r in results if r.score >= self.min_relevance_score]

        logger.info(
            f"Retrieved {len(filtered)}/{len(results)} documents "
            f"(min_score={self.min_relevance_score})"
        )

        return filtered

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved documents as context for the LLM."""
        if not results:
            return "<context>\nNo relevant documents found.\n</context>"

        context_parts = ["<context>"]
        for result in results:
            source = result.document.metadata.get("source", "unknown")
            context_parts.append(
                f"\n[Source: {source}, Relevance: {result.score:.2f}]\n{result.document.content}"
            )
        context_parts.append("\n</context>")

        return "\n".join(context_parts)

    def _format_sources(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Format source citations."""
        sources = []
        seen = set()

        for result in results:
            source = result.document.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "relevance_score": round(result.score, 3),
                    "excerpt": result.document.content[:200] + "..."
                    if len(result.document.content) > 200
                    else result.document.content,
                })

        return sources

    async def query(
        self,
        question: str,
        filter_metadata: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer a question using RAG.

        Args:
            question: User's question
            filter_metadata: Optional metadata filter for retrieval
            context: Additional context

        Returns:
            Dict with 'answer', 'sources', and 'retrieval_results'
        """
        # Retrieve relevant documents
        retrieval_results = await self.retrieve(question, filter_metadata)

        # Format context from retrieved documents
        formatted_context = self._format_context(retrieval_results)

        # Build augmented message
        augmented_message = f"{formatted_context}\n\nQuestion: {question}"

        # Generate response
        response = await self.generate_response(augmented_message, context)

        # Format result
        result = {
            "answer": response,
            "retrieval_results": retrieval_results,
        }

        if self.include_sources:
            result["sources"] = self._format_sources(retrieval_results)

        return result

    async def ingest_documents(self, documents: List[Document]) -> int:
        """Add documents to the knowledge base.

        Args:
            documents: Documents to add

        Returns:
            Number of documents added
        """
        await self.vector_store.add_documents(documents)
        return len(documents)


# ============================================================================
# Example Usage
# ============================================================================


async def main():
    """Run RAG agent example."""
    parser = argparse.ArgumentParser(description="RAG Agent Example")
    parser.add_argument(
        "--docs",
        type=str,
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stub://rag-demo",
        help="Model name (use stub:// for demo)",
    )
    parser.add_argument(
        "--persist",
        type=str,
        help="Directory to persist vector store",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Agent Example - StateSet Agents")
    print("=" * 60)

    # Initialize vector store
    try:
        vector_store = ChromaVectorStore(
            collection_name="rag_example",
            persist_directory=args.persist,
        )
        print("Using ChromaDB vector store")
    except ImportError:
        try:
            vector_store = FAISSVectorStore()
            print("Using FAISS vector store (ChromaDB not available)")
        except ImportError:
            print("ERROR: Neither ChromaDB nor FAISS available!")
            print("Install with: pip install chromadb sentence-transformers")
            return

    # Create sample documents if no directory provided
    if args.docs:
        documents = load_documents_from_directory(args.docs)
    else:
        print("\nUsing sample documents (provide --docs for custom)")
        sample_docs = [
            Document.from_text(
                """Our return policy allows customers to return any item within 30 days
                of purchase for a full refund. Items must be in original condition with
                tags attached. Electronics have a 15-day return window. Sale items are
                final sale and cannot be returned.""",
                source="policies/returns.txt",
            ),
            Document.from_text(
                """Shipping is free on orders over $50. Standard shipping takes 5-7
                business days. Express shipping (2-3 days) is available for $9.99.
                Next-day delivery is available in select areas for $19.99.""",
                source="policies/shipping.txt",
            ),
            Document.from_text(
                """Our customer support team is available Monday through Friday,
                9 AM to 6 PM EST. You can reach us by email at support@example.com,
                by phone at 1-800-EXAMPLE, or through live chat on our website.""",
                source="contact/support.txt",
            ),
            Document.from_text(
                """The StateSet Agents framework provides production-ready
                reinforcement learning for training conversational AI agents.
                Key features include GRPO/GSPO algorithms, multi-turn training,
                and distributed computing support.""",
                source="docs/overview.txt",
            ),
            Document.from_text(
                """To fine-tune an agent, first create an AgentConfig with your
                model name and training parameters. Then create a training
                environment with scenarios and reward functions. Finally,
                call the train function with your agent and environment.""",
                source="docs/training.txt",
            ),
        ]
        documents = sample_docs

    print(f"\nIndexing {len(documents)} documents...")
    await vector_store.add_documents(documents)

    # Initialize RAG agent
    config = AgentConfig(
        model_name=args.model,
        use_stub_model=args.model.startswith("stub://"),
        temperature=0.7,
        max_new_tokens=512,
    )

    agent = RAGAgent(
        config=config,
        vector_store=vector_store,
        top_k=3,
        min_relevance_score=0.2,
    )

    await agent.initialize()
    print("RAG agent initialized!")

    # Interactive query loop
    print("\n" + "=" * 60)
    print("Enter questions (type 'quit' to exit)")
    print("=" * 60)

    sample_questions = [
        "What is the return policy for electronics?",
        "How can I contact customer support?",
        "What are the shipping options?",
        "How do I train an agent with StateSet?",
    ]

    print("\nSample questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ("quit", "exit", "q"):
                break

            if not question:
                continue

            # Check for sample question number
            if question.isdigit() and 1 <= int(question) <= len(sample_questions):
                question = sample_questions[int(question) - 1]
                print(f"Using: {question}")

            # Query the RAG agent
            print("\nSearching knowledge base...")
            result = await agent.query(question)

            print("\n" + "-" * 40)
            print("ANSWER:")
            print(result["answer"])

            if result.get("sources"):
                print("\nSOURCES:")
                for source in result["sources"]:
                    print(f"  - {source['source']} (relevance: {source['relevance_score']})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
