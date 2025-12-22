import torch
from transformers import AutoModel, AutoProcessor

class Retriever:
    """
    Generic multimodal retriever wrapper.
    Supports CLIP-style models with text/image encoders.
    """

    def __init__(
        self,
        model_id: str,
        device: str = None,
        cache_dir: str = None,
        normalize: bool = True
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.normalize = normalize

        self._load_model()

    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts):
        """
        Encode a list of texts into embeddings.
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        emb = self.model.get_text_features(**inputs)

        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb

    @torch.no_grad()
    def encode_images(self, images):
        """
        Encode a list of PIL images into embeddings.
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        emb = self.model.get_image_features(**inputs)

        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb

    def score_texts(self, query_emb, doc_embs):
        """
        Cosine similarity between query and text embeddings.
        """
        return torch.matmul(query_emb, doc_embs.T)

    def score_images(self, query_emb, image_embs):
        """
        Cosine similarity between query and image embeddings.
        """
        return torch.matmul(query_emb, image_embs.T)
