import torch

class RAGModel:
    """
    Multimodal RAG wrapper that orchestrates retrieval and generation.
    """
    def __init__(
        self,
        retriever,
        generator,
        top_k_images: int = 3,
        top_k_texts: int = 3
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k_images = top_k_images
        self.top_k_texts = top_k_texts

    def retrieve(
        self,
        question: str,
        images: list,
        texts: list
    ):
        """
        Retrieve top-k image and text indices given a question.
        """
        # Embedding-based retrievers (CLIP family)
        query_emb = self.retriever.encode_text([question])
        image_embs = self.retriever.encode_images(images)
        text_embs  = self.retriever.encode_text(texts)

        image_scores = self.retriever.score_images(query_emb, image_embs)[0]
        text_scores  = self.retriever.score_texts(query_emb, text_embs)[0]

        # Ensure they're tensors (not model outputs)
        if not isinstance(image_scores, torch.Tensor):
            image_scores = torch.tensor(image_scores)
        if not isinstance(text_scores, torch.Tensor):
            text_scores = torch.tensor(text_scores)

        # Detach from computation graph if needed
        image_scores = image_scores.detach()
        text_scores = text_scores.detach()

        # Get top-k
        top_k_images = min(self.top_k_images, len(image_scores))
        top_k_texts = min(self.top_k_texts, len(text_scores))

        top_image_result = image_scores.topk(top_k_images)
        top_text_result = text_scores.topk(top_k_texts)

        top_image_indices = top_image_result.indices
        top_text_indices = top_text_result.indices

        top_image_idx = top_image_indices.cpu().tolist()
        top_text_idx  = top_text_indices.cpu().tolist()

        return top_image_idx, top_text_idx, image_scores, text_scores, top_image_indices, top_text_indices

    # def build_prompt(
    #     self,
    #     question: str,
    #     retrieved_texts: list,
    #     num_images: int
    # ) -> str:
    #     image_tokens = "<image>" * num_images
    #     context = "\n".join(f"- {t.strip()}" for t in retrieved_texts)

    #     return (
    #         f"USER: {image_tokens}\n"
    #         "Use the following information to answer the question.\n\n"
    #         f"{context}\n\n"
    #         f"Question: {question}\n\n"
    #         "Answer with ONLY the final answer. "
    #         "Do NOT include explanations, descriptions, or extra text.\n"
    #         "ASSISTANT:"
    #     )

    def build_prompt(
        self,
        question: str,
        retrieved_texts: list,
        num_images: int,
        model_type: str
    ) -> str:
        """
        Build prompt based on model type.
        """
        if model_type == "llava":
            # LLaVA uses <image> tokens
            image_tokens = "<image>" * num_images
            context = "\n".join(f"- {t.strip()}" for t in retrieved_texts if t.strip())
            return (
                f"USER: {image_tokens}\n"
                "Use the following information to answer the question.\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Answer with ONLY the final answer. "
                "Do NOT include explanations, descriptions, or extra text.\n"
                "ASSISTANT:"
            )
        
        elif model_type == "blip2":
            # BLIP2 uses the simple "Question: {} Answer:" format
            # NO image tokens, NO complex formatting
            # Context can be included as natural text before the question
            context = " ".join(t.strip() for t in retrieved_texts if t.strip())
            
            if context:
                # Include context as part of the prompt
                return f"{context} Question: {question} Answer:"
            else:
                # Simple question-answer format
                return f"Question: {question} Answer:"
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def generate(
        self,
        question: str,
        images: list,
        texts: list,
        max_new_tokens: int = 128
    ):
        """
        Full RAG forward pass.
        """
        top_image_idx, top_text_idx, image_scores, text_scores, top_image_indices, top_text_indices = self.retrieve(
            question, images, texts
        )

        top_images = [images[i] for i in top_image_idx]
        top_texts  = [texts[i] for i in top_text_idx]

        prompt = self.build_prompt(
            question=question,
            retrieved_texts=top_texts,
            num_images=len(top_images),
            model_type=self.generator.model_type 
        )

        answer = self.generator.generate(
            prompt=prompt,
            images=top_images,
            max_new_tokens=max_new_tokens
        )

        # Get selected scores using tensor indices
        selected_image_scores = image_scores[top_image_indices].detach().cpu().tolist()
        selected_text_scores = text_scores[top_text_indices].detach().cpu().tolist()

        return {
            "answer": answer,
            "retrieved_image_indices": top_image_idx,
            "retrieved_text_indices": top_text_idx,
            "image_scores": selected_image_scores,
            "text_scores": selected_text_scores,
        }