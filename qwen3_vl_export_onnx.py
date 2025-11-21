import os
import shutil
import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel
from transformers import AutoProcessor
from typing import Any, Callable, Optional, Union

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class ArgsConfig:
    """Configuration for Qwen3-VL model export ONNX"""

    # Model parameters
    qwen_path: str = 'weights/qwen3-vl-2b'
    """Path to the qwen directory or directories"""

    onnx_path: str = 'onnx_qwen3_vl'
    """Directory to save onnx model checkpoints."""

    batch_size: int = 1
    """Batch size of input for ONNX model inference"""

    imgs_nums: int = 1
    """Number of images for ONNX model inference"""


class Qwen3VLTextModelOpt(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # if cache_position is None:
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        text_position_ids = position_ids[0]

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # create attentions_mask by eager kind
        base_mask = torch.tril(torch.ones(position_ids.size(2), position_ids.size(2), dtype=torch.bool)).to(inputs_embeds.device)
        mask = torch.full((position_ids.size(2), position_ids.size(2)), fill_value=-3.4028e+38, dtype=torch.float32, device=inputs_embeds.device)
        mask = mask.masked_fill(base_mask, 0.0)

        # size from 144*144 -> 1*1*144*144
        attention_mask = mask.unsqueeze(0).unsqueeze(0)


        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _deepstack_process(
            self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        indices = torch.nonzero(visual_pos_masks.squeeze(0), as_tuple=False).squeeze(1)
        indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(1, -1, hidden_states.size(2))
        new_values = hidden_states[0, indices] + visual_embeds
        hidden_states = hidden_states.scatter(1, indices_expanded, new_values.unsqueeze(0))
        return hidden_states


class Qwen3VLVisualModelOpt(Qwen3VLModel):
    def __init__(self, qwen_config, onnx_config):
        self.batch_size = onnx_config.batch_size
        self.imgs_nums = onnx_config.imgs_nums
        super().__init__(qwen_config)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        mrope_position_deltas = []
        total_input_ids = input_ids

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i in range(self.batch_size):
            input_ids = input_ids[i]
            image_nums = torch.tensor([self.imgs_nums], dtype=torch.int64)
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            st_idx = 0
            remain_images = image_nums
            for _ in range(image_nums):
                ed_image = input_tokens.index(image_token_id, st)

                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                st_idx = llm_pos_ids_list[-1].max() + 1

            text_len = input_ids.shape[0] - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            inputs_embeds = self.get_input_embeddings()(input_ids) # torch.Size([1, 144, 2048])


            # process image use vit model
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = torch.stack(deepstack_image_embeds)

            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding

            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                attention_mask=attention_mask_tensor,
            )

            return position_ids, attention_mask, inputs_embeds, visual_pos_masks, deepstack_visual_embeds


class Qwen3VLForConditionalGenerationOpt(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            hidden_states: torch.LongTensor = None,
            **kwargs,
        ):
        slice_indices = hidden_states.size(1)
        logits = self.lm_head(hidden_states[:, slice_indices - 1, :])
        return logits



def export_qwen_llm(qwen_model, inputs, onnx_path, config):
    # Remove and create new onnx dir
    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    # qwen_model.config._attn_implementation = "eager"
    # Create Text Model
    model = Qwen3VLTextModelOpt(qwen_model.config).to(config.device)
    model.load_state_dict(qwen_model.state_dict())
    model.eval()

    # Create ONNX inputs
    input_ids = inputs["input_ids"]
    batch_size, seq_len = input_ids.shape
    deepstack_visual_len = 3

    position_ids = torch.ones((3, batch_size, seq_len), dtype=torch.int64).to(config.device) # torch.Size([3, 1, 144])
    inputs_embeds = torch.zeros((batch_size, seq_len, 2048), dtype=torch.float32).to(config.device) # torch.Size([1, 144, 2048])

    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()
    visual_pos_masks = visual_pos_masks.to(config.device) # torch.Size([1, 144])
    deepstack_visual_embeds = torch.randn((deepstack_visual_len, x, 2048), dtype=torch.float32).to(config.device) # torch.Size([3, 67, 2048])
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).to(config.device) # mask allways be one, so discard it

    torch.onnx.export(
        model,
        (position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds),
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["position_ids", "inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds"],
        output_names=["hidden_states"],
        dynamic_axes={
            "position_ids": {1: "batch_size", 2: "seq_length"},
            "inputs_embeds": {0: "batch_size", 1: "seq_length"},
            "visual_pos_masks": {0: "batch_size", 1: "seq_length"},
            "deepstack_visual_embeds": {1: "visual_seqlen"},
            "hidden_states": {0: "batch_size", 1: "seq_length"},
        },
        verbose=True,
    )

    print("Export Qwen3 LLM done!")
    del position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds, attention_mask
    del model

def export_qwen_vit(
        qwen_model,
        inputs,
        onnx_path,
        config,
):

    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    model = Qwen3VLVisualModelOpt(qwen_model.config, config).to(config.device)
    model.load_state_dict(qwen_model.state_dict())
    model = model.to(config.device)
    model.eval()

    input_ids = inputs["input_ids"].clone()    # shape torch.Size([1, 144])
    attention_masks = inputs["attention_mask"].clone()    # shape torch.Size([1, 144])
    pixel_values = inputs["pixel_values"].clone()    # shape torch.Size([512, 1536])
    image_grid_thw = inputs["image_grid_thw"].clone()    # shape torch.Size([2, 3])


    torch.onnx.export(
        model,
        (input_ids, attention_masks, pixel_values, image_grid_thw),
        onnx_path,
        input_names=["input_ids", "attention_masks", "pixel_values", "image_grid_thw"],
        output_names=["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_masks": {0: "batch_size", 1: "seq_length"},
            "pixel_values": {0: "img_length"},
            "image_grid_thw": {0: "num_images"},
        },
        verbose=True,
    )

    print("Export Qwen3 Vit done!")
    del input_ids, attention_masks, pixel_values, image_grid_thw
    del model


def export_qwen_vlm(qwen_model, inputs, onnx_path, config):
    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    model = Qwen3VLForConditionalGenerationOpt(qwen_model.config)
    model.load_state_dict(qwen_model.state_dict())
    model = model.to(config.device)

    input_ids = inputs["input_ids"]
    batch_size, seq_len = input_ids.shape
    hidden_states = torch.randn((batch_size, seq_len, 2048), dtype=torch.float32).to(config.device)
    torch.onnx.export(
        model,
        (hidden_states,),
        onnx_path,
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "seq_length"},
        },
        verbose=True,
    )

    print("Export Qwen3 Generate done!")
    del hidden_states
    del model


def run_export(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    torch.manual_seed(42)

    model_input = get_model_input(config)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.float32, device_map='cpu', attn_implementation="eager")
    print("Init model load done!")
    # with torch.no_grad():
    #     model_output = qwen_model(**model_input, output_hidden_states=True, use_cache=False, return_dict=True)

    export_qwen_llm(
        qwen_model=qwen_model.model.language_model,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'llm/llm.onnx'),
        config=config
    )
    export_qwen_vit(
        qwen_model=qwen_model.model,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'vit/vit.onnx'),
        config=config,
    )
    export_qwen_vlm(
        qwen_model=qwen_model,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'vlm/vlm.onnx'),
        config=config,
    )



def get_model_input(config):
    processor = AutoProcessor.from_pretrained(config.qwen_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "input1.png",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Check context
    assert len(messages) == config.batch_size, f"messages number should be equal to batch_size, but now messages batch size = {len(messages)}, config batch_size = {config.batch_size}"
    for i in range(config.batch_size):
        numbers_img = sum([1 if content["type"] == "image" else 0 for content in messages[i]["content"]])
        assert numbers_img == config.imgs_nums, f"The number of imgs is mismatch, {numbers_img} != {config.imgs_nums}"

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(config.device)


if __name__ == "__main__":
    config = ArgsConfig()
    run_export(config)