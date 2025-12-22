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
        Retrieve top-k images and texts given a question.
        """
        # Encode query
        query_emb = self.retriever.encode_text([question])

        # Encode candidates
        image_embs = self.retriever.encode_images(images)
        text_embs  = self.retriever.encode_text(texts)

        # Score
        image_scores = self.retriever.score_images(query_emb, image_embs)[0]
        text_scores  = self.retriever.score_texts(query_emb, text_embs)[0]

        # Rank
        top_image_idx = image_scores.topk(self.top_k_images).indices.tolist()
        top_text_idx  = text_scores.topk(self.top_k_texts).indices.tolist()

        return (
            [images[i] for i in top_image_idx],
            [texts[i] for i in top_text_idx],
            image_scores[top_image_idx],
            text_scores[top_text_idx],
        )
    
    
    def build_prompt(
        self,
        question: str,
        retrieved_texts: list,
        num_images: int
    ) -> str:
        """
        Build a multimodal prompt for the generator.
        """

        image_tokens = "<image>" * num_images

        context = "\n".join(
            f"- {t.strip()}" for t in retrieved_texts
        )

        prompt = (
            f"USER: {image_tokens}\n"
            "Use the following information to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with ONLY the final answer. "
            "Do NOT include explanations, descriptions, or extra text.\n"
            "ASSISTANT:"
        )

        return prompt



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
        top_images, top_texts, image_scores, text_scores = self.retrieve(
            question, images, texts
        )

        prompt = self.build_prompt(
            question=question,
            retrieved_texts=top_texts,
            num_images=len(top_images)
        )

        output = self.generator.generate(
            prompt=prompt,
            images=top_images,
            max_new_tokens=max_new_tokens
        )

        return {
            "answer": output,
            "retrieved_images": top_images,
            "retrieved_texts": top_texts,
            "image_scores": image_scores,
            "text_scores": text_scores,
        }
