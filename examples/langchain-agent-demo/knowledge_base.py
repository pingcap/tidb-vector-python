from typing import Union, List, Callable
from dotenv import load_dotenv, find_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from functools import partial
import os
from tidb_vector.integrations import TiDBVectorClient
from langchain_core.retrievers import BaseRetriever

# load environment variables
_ = load_dotenv(find_dotenv())

# https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-encoding_format
Vector = Union[List[float], List[int]]
Vectors = List[Vector]


def sentence_transformer_embedding_function(
    embed_model: SentenceTransformer, sentences: Union[str, List[str]]
) -> Union[Vector, Vectors]:
    """
    Generates vector embeddings for the given text using the sentence-transformers model.

    Args:
        embed_model (SentenceTransformer): The sentence-transformers model to use.
        sentences (Union[str, List[str]]): The text or list of texts for which to generate embeddings.

    Returns:
        if sentences is a single string:
            List[float]: The embedding for the input sentence.
        if sentences is a list of strings:
            List[List[float]]: The embeddings for the input sentences.


    Examples:
        Below is a code snippet that shows how to use this function:
        ```python
        embeddings = sentence_transformer_embedding_function("Hello, world!")
        ```
        or
        ```python
        embeddings = sentence_transformer_embedding_function(["Hello, world!"])
        ```
    """

    return embed_model.encode(sentences).tolist()


class TiRetriever(BaseRetriever):
    """A retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """

    """Vector Database Client. For example, TiDBVectorClient."""
    rm: TiDBVectorClient
    """"""
    embedding_function: Callable[[str], Vector]
    """The number of top documents to return."""
    k: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Sync implementations for retriever."""
        query_embeddings = self.embedding_function(str(query))
        tidb_vector_res = self.rm.query(query_embeddings, k=self.k)
        passages_scores = {}
        for res in tidb_vector_res:
            passages_scores[res.document] = res.distance
        sorted_passages = sorted(passages_scores.items(), key=lambda x: x[1], reverse=True)
        return [Document(text) for (text, score) in sorted_passages]


embed_model = SentenceTransformer(os.environ.get('SENTENCE_TRANSFORMERS_MODEL'), trust_remote_code=True)
embed_model_dim = embed_model.get_sentence_embedding_dimension()

embedding_function = partial(sentence_transformer_embedding_function, embed_model)

tidb_vector_client = TiDBVectorClient(
    table_name=os.environ.get('TIDB_TABLE_NAME', 'embedded_documents'),
    connection_string=os.environ.get('TIDB_DATABASE_URL'),
    vector_dimension=embed_model_dim,
    drop_existing_table=True,
)

print("Connected to TiDB.")
print("describe table:")
print(tidb_vector_client.execute("describe embedded_documents;"))

print("Initializing the retriever...")
retriever = TiRetriever(rm=tidb_vector_client, embedding_function=embedding_function, k=3)
print("Retriever initialized successfully.")

print("Loading sample data...")
# test sample data
# load sample_data.txt  if not local file, you can use requests.get(url).text
# sample data url: https://raw.githubusercontent.com/wxywb/dspy_dataset_sample/master/sample_data.txt
with open('sample_data.txt', 'r') as f:
    # I prepare a small set of data for speeding up embedding, you can replace it with your own data.
    print("sample_data.txt found.")
    sample_data = f.read()
print("Sample data loaded successfully.")

print("Embedding sample data...")
documents = []
for idx, passage in enumerate(sample_data.split('\n')[:3]):
    embedding = embedding_function([passage])[0]
    print(idx, passage[:10], embedding[:5])
    if len(passage) == 0:
        continue
    documents.append(
        {
            "id": str(idx),
            "text": passage,
            "embedding": embedding,
            "metadata": {"category": "album"},
        }
    )
print("Sample data embedded successfully.")
print("Sample data number:", len(documents))

print("Inserting documents into TiDB...")
tidb_vector_client.insert(
    ids=[doc["id"] for doc in documents],
    texts=[doc["text"] for doc in documents],
    embeddings=[doc["embedding"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents],
)
print("Documents inserted successfully.")

print("# ---- Init Finish ----")
