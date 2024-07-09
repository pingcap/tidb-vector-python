from typing import List, Optional, Union
import dspy
from dsp.utils import dotdict
from sentence_transformers import SentenceTransformer

from tidb_vector.integrations import TiDBVectorClient


def sentence_transformer_embedding_function(model: str, sentences: Union[str, List[str]]):
    """
    Generates vector embeddings for the given text using the sentence-transformers model.

    Args:
        model (str): The name or path of the sentence-transformers model to use.
        sentences (List[str]): A list of text sentences for which to generate embeddings.

    Returns:
        List: A list of embeddings for the given text sentences.

    Examples:
        Below is a code snippet that shows how to use this function:
        ```python
        embeddings = sentence_transformer_embedding_function("sentence-transformers/multi-qa-mpnet-base-dot-v1", ["Hello, world!"])
        ```
    """
    embed_model = SentenceTransformer(model, trust_remote_code=True)
    return embed_model.encode(sentences)


class TidbRM(dspy.Retrieve):
    """
    A retrieval module that uses TiDBVectorClient to return passages for a given query.

    Args:
        tidb_vector_client (TiDBVectorClient): The TiDBVectorClient instance to use for querying TiDB.
        embedding_function (callable): The function to convert a list of text to embeddings.
            The embedding function should take a list of text strings as input and output a list of embeddings.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        use OpenAI
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = TidbRM(
            tidb_vector_client=tidb_vector_client,
            embedding_function=sentence_transformer_embedding_function
        )
        dspy.settings.configure(rm=retriever_model)
        ```

        use Ollama
        ```python
        llm = dspy.OllamaLocal(model="llama3:8b")
        retriever_model = TidbRM(
            tidb_vector_client=tidb_vector_client,
            embedding_function=llm
        )

    """

    def __init__(self, tidb_vector_client: TiDBVectorClient, embedding_function: Optional[callable] = None, k: int = 3):
        super().__init__(k)
        self.tidb_vector_client = tidb_vector_client
        self.embedding_function = embedding_function
        self.k = k

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction:
        """
        Retrieve passages for the given query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries for which to retrieve passages.
            k (Optional[int]): The number of top passages to retrieve. Defaults to 3.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.

        Examples:
            Below is a code snippet that shows how to use this function:
            ```python
            passages = self.retrieve("Hello, world!")
            ```
        """
        if self.embedding_function is None:
            raise ValueError("embedding_function is required to use TidbRM")

        query_embeddings = self.embedding_function(query_or_queries)
        tidb_vector_res = self.tidb_vector_client.query(query_vector=query_embeddings, k=k)
        passages_scores = {}
        for res in tidb_vector_res:
            res.metadata = dotdict(res.metadata)
            passages_scores[res.document] = res.distance
        sorted_passages = sorted(passages_scores.items(), key=lambda x: x[1], reverse=True)
        k_sorted_passages = sorted_passages[:k]

        return dspy.Prediction(passages=[dotdict({"long_text": passage}) for passage, _ in k_sorted_passages])


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, rm):
        super().__init__()
        self.retrieve = rm

        # This signature indicates the task imposed on the COT module.
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # Use milvus_rm to retrieve context for the question.
        context = self.retrieve(question).passages
        # COT module takes "context, query" and output "answer".
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=[item.long_text for item in context], answer=prediction.answer)
