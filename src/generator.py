import torch
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration

class Generator:
    """
    Generic multimodal generator for RAG.
    Supports LLaVA-style and other vision-language chat models.
    """
    def __init__(
            self,
            model_id: str,
            device: str = None,
            cache_dir: str = None,
            dtype = torch.float16,
            trust_remote_code: bool = True
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.torch_dtype = dtype
        self.trust_remote_code = trust_remote_code

        self._load_model()

    def _load_model(self):
        """
        Load processor and model depending on architecture.
        """

        # Processor (handles text + image formatting)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code
        )

        if "llava" in self.model_id.lower():
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
                device_map="auto"
            )
        else:
            # Generic fallback (e.g., Qwen-VL-Chat)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
                device_map="auto"
            )

        self.model.eval()

    def generate(
        self,
        prompt: str,
        images=None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7
    ):
        """
        Generate a response given a prompt and optional images.
        """

        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None
            )

        return self.processor.decode(output_ids[0], skip_special_tokens=True)