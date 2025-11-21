import torch
import numpy as np
import onnxruntime as ort
from typing import Optional, Union, List
import os
from transformers import AutoProcessor


class QwenVLDecoder:
    def __init__(self, onnx_path: str, max_length: int = 512, eos_token_id: int = 151643):
        """
        Args:
            onnx_path: Path to the ONNX model file
            max_length: Maximum generation length
            eos_token_id: End-of-sequence token ID
        """
        self._load_session(onnx_path)
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def _load_session(self, onnx_path: str):
        onnx_qwen_vit = os.path.join(onnx_path, 'vit/vit.onnx')
        onnx_qwen_llm = os.path.join(onnx_path, 'llm/llm.onnx')
        onnx_qwen_vlm = os.path.join(onnx_path, 'vlm/vlm.onnx')
        self.vit_session = ort.InferenceSession(
                                            onnx_qwen_vit,
                                            providers=["CUDAExecutionProvider"]
                                            )
        self.llm_session = ort.InferenceSession(
                                            onnx_qwen_llm,
                                            providers=["CUDAExecutionProvider"]
                                            )
        self.vlm_session = ort.InferenceSession(
                                            onnx_qwen_vlm,
                                            providers=["CUDAExecutionProvider"]
                                            )

    def _onnx_inference(self, inputs: dict[str: torch.Tensor]) -> np.ndarray:
        # Vit process
        vit_outputs = self.vit_session.run(
            input_feed=inputs,
            output_names=["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks",
                          "deepstack_visual_embeds"],
        )

        # LLM process
        llm_inputs = {
            "position_ids": vit_outputs[0],
            "inputs_embeds": vit_outputs[2],
            "visual_pos_masks": vit_outputs[3],
            "deepstack_visual_embeds": vit_outputs[4]
        }
        llm_outputs = self.llm_session.run(
            input_feed=llm_inputs,
            output_names=["hidden_states"],
        )

        # VLM process
        vlm_inputs = {
            "hidden_states": llm_outputs[0],
        }
        vlm_outputs = self.vlm_session.run(
            input_feed=vlm_inputs,
            output_names=["logits"],
        )

        return vlm_outputs[0] # (batch_size, seq_len)


    def logits_to_tokens(self, last_token_logits: np.ndarray, method: str = "greedy",
                         temperature: float = 1.0, top_k: Optional[int] = None) -> np.ndarray:
        """
        Convert logits to tokens using specified decoding method.

        Args:
            last_token_logits: ONNX Model output logits of shape (batch_size, vocab_size)
            method: Decoding method - "greedy" or "sampling"
            temperature: Temperature for sampling
            top_k: Top-k filtering parameter

        Returns:
            Array of token IDs of shape (batch_size, 1)
        """

        if method == "greedy":
            # Greedy decoding - select token with highest probability
            next_tokens = np.argmax(last_token_logits, axis=-1, keepdims=True)

        elif method == "sampling":
            # Apply temperature and softmax
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None and top_k > 0:
                # Get top-k indices and values
                indices_to_remove = last_token_logits < np.partition(
                    last_token_logits, -top_k, axis=-1)[..., -top_k, None]
                last_token_logits[indices_to_remove] = -float('inf')

            # Apply softmax to get probabilities
            probabilities = self._softmax(last_token_logits)

            # Sample from the probability distribution
            next_tokens = np.array([
                np.random.choice(len(probs), p=probs)
                for probs in probabilities
            ]).reshape(-1, 1)

        else:
            raise ValueError(f"Unknown decoding method: {method}")

        return next_tokens.astype(np.int64)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _prepare_inputs(self, input_ids: np.ndarray,
                       attention_mask: np.ndarray,
                       pixel_values: np.ndarray,
                       image_grid_thw: np.ndarray,
    ) -> dict:
        """Prepare inputs for ONNX model inference"""
        inputs = {
            'input_ids': input_ids,
            'attention_masks': attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        return inputs

    def generate(self,
                 input_ids: np.ndarray = None,
                 attention_masks: np.ndarray = None,
                 pixel_values: np.ndarray = None,
                 image_grid_thw: np.ndarray = None,
                 max_new_tokens: Optional[int] = None,
                 method: str = "greedy",
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 stop_token_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial input token IDs
            attention_masks: Attention mask for input tokens
            pixel_values: Image features for multimodal input
            image_grid_thw: Image features for multimodal input
            max_new_tokens: Maximum number of new tokens to generate
            method: Decoding method
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            stop_token_ids: Additional token IDs that trigger generation stop

        Returns:
            Generated token IDs
        """
        # Set stop token IDs
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.eos_token_id)

        # Set maximum generation length
        max_new_tokens = min(max_new_tokens or self.max_length, self.max_length)

        # Initialize generation
        current_input_ids = input_ids.copy()
        current_attention_mask = attention_masks.copy()
        all_generated_tokens = []

        for step in range(max_new_tokens):
            # Prepare inputs for ONNX model
            inputs = self._prepare_inputs(current_input_ids, current_attention_mask, pixel_values, image_grid_thw)

            # Run ONNX model inference
            last_logits = self._onnx_inference(inputs)

            # Decode next tokens
            next_tokens = self.logits_to_tokens(
                last_logits, method=method, temperature=temperature, top_k=top_k
            )

            # Update sequences
            current_input_ids = np.concatenate([current_input_ids, next_tokens], axis=1)
            current_attention_mask = np.concatenate([
                current_attention_mask,
                np.ones((current_input_ids.shape[0], 1), dtype=np.int64)
            ], axis=1)

            # Store generated tokens
            all_generated_tokens.append(next_tokens)

            # Check stopping condition
            if any(np.isin(next_tokens.flatten(), stop_token_ids)):
                break

        # Return all generated tokens
        return np.concatenate(all_generated_tokens, axis=1) if all_generated_tokens else np.array([])

    def batch_generate(self,
                       input_ids_list: List[np.ndarray],
                       attention_mask_list: Optional[List[np.ndarray]] = None,
                       image_features_list: Optional[List[np.ndarray]] = None,
                       **kwargs) -> List[np.ndarray]:
        """
        Generate tokens for multiple inputs in batch.

        Args:
            input_ids_list: List of input token arrays
            attention_mask_list: List of attention mask arrays
            image_features_list: List of image feature arrays

        Returns:
            List of generated token arrays
        """
        results = []
        for i, input_ids in enumerate(input_ids_list):
            attention_mask = (attention_mask_list[i]
                              if attention_mask_list and i < len(attention_mask_list)
                              else None)
            image_features = (image_features_list[i]
                              if image_features_list and i < len(image_features_list)
                              else None)

            result = self.generate(input_ids, attention_mask, image_features, **kwargs)
            results.append(result)

        return results


def chat_inputs(processor):
    template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "demo_data/input1.png",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    tensor_inputs = processor.apply_chat_template(
        template,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        "input_ids": tensor_inputs["input_ids"].cpu().numpy(),
        "attention_masks": tensor_inputs["attention_mask"].cpu().numpy(),
        "pixel_values": tensor_inputs["pixel_values"].cpu().numpy(),  # torch.int64 → np.int64
        "image_grid_thw": tensor_inputs["image_grid_thw"].cpu().numpy(),  # torch.float32 → np.float32
    }

    return inputs

# Usage example
def main():
    qwen_path = 'weights/qwen3-vl-2b'
    onnx_path = 'onnx_qwen3_vl'

    # Example input (adjust based on your actual model inputs)
    processor = AutoProcessor.from_pretrained(qwen_path)
    messages = chat_inputs(processor)

    # Initialize decoder
    decoder = QwenVLDecoder(onnx_path)


    # Generate tokens
    generated_tokens = decoder.generate(
        **messages,
        max_new_tokens=1280,
        method="sampling",  # or "sampling"
        temperature=0.8,
        top_k=20
    )
    output_text = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("Generated tokens shape:", generated_tokens.shape)
    print("Generated tokens:", generated_tokens)
    print("Generated text:", output_text)


if __name__ == "__main__":
    main()