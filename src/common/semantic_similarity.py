import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SemanticSimilarityScorer:
    def __init__(
        self,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    def compute_similarities(self, query: str, docs: list[str]) -> list[float]:
        passages = [f"passage: {doc}" for doc in docs]
        query = f"query: {query}"

        # Encoe the query
        query_embed = self.encode([query])[0]  # shape (hidden_dim,)

        scores = []
        for i in range(0, len(passages), self.batch_size):
            embeddings = self.encode(passages[i : i + self.batch_size])
            scores += (query_embed @ embeddings.T).tolist()
        return scores

    @torch.no_grad()
    def encode(
        self, inputs: list[str], normalize_embeddings: bool = True
    ) -> torch.Tensor:
        # Tokenize input texts
        input_dict = self.tokenizer(
            inputs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Get the embeddings
        outputs = self.model(**input_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, input_dict["attention_mask"]
        )
        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
