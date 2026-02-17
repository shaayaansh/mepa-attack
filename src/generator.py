import torch
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, Blip2ForConditionalGeneration
from transformers import (
    AutoProcessor,
    InstructBlipForConditionalGeneration,
)

class Generator:
    """
    Generic multimodal generator for RAG.
    Supports LLaVA-style and other vision-language chat models.
    """
    def __init__(
            self,
            model_type: str,  # "llava", "qwen-vl", "clip"
            model_id: str,
            device: str = None,
            cache_dir: str = None,
            dtype = torch.float16,
            trust_remote_code: bool = True
    ):
        self.model_type = model_type    
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.torch_dtype = dtype
        self.trust_remote_code = trust_remote_code

        self._load_model()

    def _load_model(self):
        if self.model_type == "llava":
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                use_fast=False,
                trust_remote_code=True
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                device_map="auto"
            )

        elif self.model_type == "blip2":
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )

            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                device_map="auto"
            )

        else:
            raise ValueError(f"Unknown generator type: {self.model_type}")

        if self.model is not None:
            self.model.eval()

    def generate(
        self,
        prompt: str,
        images=None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7
    ):

        if self.model_type == "llava":
            return self._generate_llava(prompt, images, max_new_tokens, do_sample, temperature)

        elif self.model_type == "blip2":
            return self._generate_blip2(prompt, images, max_new_tokens)

        else:
            raise ValueError(f"Unknown generator type: {self.model_type}")

    def _generate_llava(self, prompt, images, max_new_tokens, do_sample, temperature):
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


    def _generate_blip2(self, prompt, images, max_new_tokens):
        # FIX: Ensure images is always a list
        if images is not None and not isinstance(images, list):
            images = [images]
        
        inputs = self.processor(
            images=images,   
            text=prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        
        answer = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]
        
        return answer
