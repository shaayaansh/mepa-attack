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
        query_emb = self.retriever.encode_text([question])

        image_embs = self.retriever.encode_images(images)
        text_embs  = self.retriever.encode_text(texts)

        image_scores = self.retriever.score_images(query_emb, image_embs)[0]
        text_scores  = self.retriever.score_texts(query_emb, text_embs)[0]

        top_image_idx = image_scores.topk(self.top_k_images).indices.tolist()
        top_text_idx  = text_scores.topk(self.top_k_texts).indices.tolist()

        return top_image_idx, top_text_idx, image_scores, text_scores

    def build_prompt(
        self,
        question: str,
        retrieved_texts: list,
        num_images: int
    ) -> str:

        image_tokens = "<image>" * num_images

        context = "\n".join(f"- {t.strip()}" for t in retrieved_texts)

        return (
            f"USER: {image_tokens}\n"
            "Use the following information to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with ONLY the final answer. "
            "Do NOT include explanations, descriptions, or extra text.\n"
            "ASSISTANT:"
        )

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
        top_image_idx, top_text_idx, image_scores, text_scores = self.retrieve(
            question, images, texts
        )

        top_images = [images[i] for i in top_image_idx]
        top_texts  = [texts[i] for i in top_text_idx]

        prompt = self.build_prompt(
            question=question,
            retrieved_texts=top_texts,
            num_images=len(top_images)
        )

        answer = self.generator.generate(
            prompt=prompt,
            images=top_images,
            max_new_tokens=max_new_tokens
        )

        return {
            "answer": answer,
            "retrieved_image_indices": top_image_idx,
            "retrieved_text_indices": top_text_idx,
            "image_scores": image_scores[top_image_idx].tolist(),
            "text_scores": text_scores[top_text_idx].tolist(),
        }
