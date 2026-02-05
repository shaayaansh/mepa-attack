import torch
from transformers import AutoModel, AutoProcessor

class Retriever:
    """
    Generic multimodal retriever wrapper.
    Supports CLIP-style models with text/image encoders.
    """
    def __init__(
        self,
        model_type: str,  # "clip", "openclip", "sigclip", "blip2"
        model_id: str,
        device: str = None,
        cache_dir: str = None,
        normalize: bool = True
    ):
        self.model_type = model_type
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.normalize = normalize
        self._load_model()

    def _load_model(self):
        if self.model_type in ["clip", "openclip", "sigclip"]:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                use_safetensors=True
            ).to(self.device)
            self.model.eval()

        elif self.model_type == "flava":
            from transformers import FlavaProcessor, FlavaModel
            self.processor = FlavaProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            self.model = FlavaModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()

        else:
            raise ValueError(f"Unknown retriever type: {self.model_type}")

    def _extract_embedding(self, output):
        """
        Extract tensor embedding from model output.
        Handles both raw tensors and BaseModelOutput objects.
        """
        if isinstance(output, torch.Tensor):
            return output
        elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state[:, 0]  # Use CLS token
        else:
            raise ValueError(f"Cannot extract embedding from output type: {type(output)}")


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
        )
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.model_type in ["clip", "openclip", "sigclip"]:
            output = self.model.get_text_features(**inputs)
            emb = self._extract_embedding(output)
        elif self.model_type == "flava":
            # FLAVA returns text embeddings
            outputs = self.model.get_text_features(**inputs)
            # First extract the tensor from the output object
            emb = self._extract_embedding(outputs)
            # Then check if it's a sequence and take CLS token if needed
            if emb.dim() == 3:  # [batch, seq_len, hidden_dim]
                emb = emb[:, 0, :]  # Take CLS token (first token)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if self.normalize:
            emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        
        return emb

    @torch.no_grad()
    def encode_images(self, images):
        """
        Encode a list of PIL images into embeddings.
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.model_type in ["clip", "openclip", "sigclip"]:
            output = self.model.get_image_features(**inputs)
            emb = self._extract_embedding(output)
        elif self.model_type == "flava":
            # FLAVA returns image embeddings
            outputs = self.model.get_image_features(**inputs)
            # First extract the tensor from the output object
            emb = self._extract_embedding(outputs)
            # Then check if it's a sequence and take CLS token if needed
            if emb.dim() == 3:  # [batch, seq_len, hidden_dim]
                emb = emb[:, 0, :]  # Take CLS token (first token)
        elif self.model_type == "blip2":
            # BLIP2 doesn't have get_image_features, use vision_model directly
            outputs = self.model.vision_model(**inputs)
            emb = self._extract_embedding(outputs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if self.normalize:
            emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        
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


    @torch.no_grad()
    def score_image_text_pairs(self, images, texts):
        """
        BLIP-2 image-text matching scores.
        Returns a [num_images, num_texts] score matrix.
        """
        assert self.model_type == "blip2"
        
        scores = torch.zeros(len(images), len(texts), device=self.device)
        
        for i, img in enumerate(images):
            for j, txt in enumerate(texts):
                try:
                    inputs = self.processor(
                        images=img,
                        text=txt,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Debug: print what's in inputs
                    print(f"\n=== Debug for image {i}, text {j} ===")
                    for k, v in inputs.items():
                        print(f"Key: {k}, Type: {type(v)}, Value type: {type(v) if not torch.is_tensor(v) else 'Tensor'}")
                    
                    # Move only tensor inputs to device
                    inputs = {
                        k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in inputs.items()
                    }
                    
                    print("About to call model...")
                    outputs = self.model(**inputs)
                    print("Model call successful")
                    
                    # Get the ITM (image-text matching) score
                    if hasattr(outputs, 'itm_score'):
                        score = outputs.itm_score
                    elif hasattr(outputs, 'logits'):
                        score = outputs.logits[:, 1] if outputs.logits.dim() > 1 else outputs.logits
                    else:
                        raise ValueError("Cannot find ITM score in model outputs")
                    
                    scores[i, j] = score.squeeze()
                    
                except Exception as e:
                    print(f"\nERROR at image {i}, text {j}:")
                    print(f"Error type: {type(e)}")
                    print(f"Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        return scores