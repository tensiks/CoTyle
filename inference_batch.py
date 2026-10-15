import os
import copy
import torch
from PIL import Image
from tqdm import tqdm
from models.pipe import CoTylePipeline, PiCoTylePipeline

from io import BytesIO
import requests
from models.vlm_unitok import UniTok
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)
import argparse
from models.utils import (
    set_seed,
    load_and_process_config,
    patched_from_model_config,
    concatenate_images_with_sref,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.model import StyleGenerator
from models.style_code import add_style_code_args, make_style_ids_fn
import json
from models.model import Qwen2_5_VLForConditionalGeneration_Quant, Qwen2_5_VL_Quant
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)
from transformers.generation.configuration_utils import GenerationConfig

_original_from_model_config = GenerationConfig.from_model_config
GenerationConfig.from_model_config = classmethod(patched_from_model_config)


def main(args):
    output_dir = args.output_path

    unitok_config = {
        "unitok_embed_dim": 3584,
        "unitok_vocab_width": 64,
        "unitok_vocab_size": 1024,
        "unitok_e_temp": 0.01,
        "unitok_num_codebooks": 1,
        "unitok_le": 0.0,
    }
    weight_type = torch.bfloat16

    style_generator_path = os.path.join(args.model_path, "prior")
    config = AutoConfig.from_pretrained(f"{style_generator_path}/config.json")
    style_generator = StyleGenerator._from_config(config)
    state_dict = torch.load(f"{style_generator_path}/prior.pth", map_location="cpu")
    style_generator.load_state_dict(state_dict)
    style_generator.to("cuda", dtype=weight_type)

    # loading codebook
    unitok = UniTok(unitok_config)
    unitok_state_dict = torch.load(
        f"{args.model_path}/codebook/model.pth", map_location="cpu"
    )
    unitok.load_state_dict(unitok_state_dict)
    unitok.to("cuda", dtype=weight_type)

    # loading text_encoder
    if args.accelerate:
        pipeline = PiCoTylePipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            text_encoder=None,
            processor=None,
        )
    else:
        pipeline = CoTylePipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            text_encoder=None,
            processor=None,
        )

    qwen_text_visual_encoder = Qwen2_5_VLForConditionalGeneration_Quant.from_pretrained(
        os.path.join(args.model_path, "text_encoder"),
        torch_dtype=weight_type,
        # low_cpu_mem_usage=True,
        device_map="cuda",
    )
    #   Do text encoding here
    qwen_text_visual_encoder = Qwen2_5_VL_Quant(unitok, qwen_text_visual_encoder)
    qwen_text_visual_encoder.to("cuda", dtype=weight_type)

    pipeline.text_encoder = qwen_text_visual_encoder
    processor = Qwen2VLProcessor.from_pretrained(
        os.path.join(args.model_path, "processor"),
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    pipeline.processor = processor

    if args.accelerate:
        adapter_name = pipeline.load_piflow_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
            "Lakonik/pi-Qwen-Image",
            subfolder="gmqwen_k8_piid_4step",
            target_module_name="transformer",
        )
        pipeline.scheduler = (
            FlowMatchEulerDiscreteScheduler.from_config(  # use fixed shift=3.2
                pipeline.scheduler.config,
                shift=3.2,
                shift_terminal=None,
                use_dynamic_shifting=False,
            )
        )

    del qwen_text_visual_encoder

    # pipeline.to("cpu", dtype=torch.bfloat16)
    # pipeline.enable_model_cpu_offload()
    # Same settings as inference.py: leaf_level offload re-creates the weights in RAM on every
    # offload and gets the process OOM-killed when the 41 GB transformer does not fit in RAM.
    pipeline.transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=16,
        use_stream=True,
        low_cpu_mem_usage=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    os.makedirs(output_dir, exist_ok=True)
    placeholder_image = Image.new("RGB", (392, 392), (0, 0, 0))
    with open(f"{args.model_path}/freq.json", "r") as f:
        code_freq = json.load(f)

    prompts = []
    with open(args.prompt_file_path, "r") as f:
        for line in f:
            prompts.append(line.strip())

    negative_prompt = "丑陋，怪物，怪兽，畸形，变异，结构不合理，肢体不合理，人脸扭曲, 肢体错乱,突兀"
    true_cfg_scale = 6.0

    def image_path(style_code, prompt_i):
        return f"{output_dir}/batch/{style_code}-{prompt_i}.png"

    os.makedirs(f"{output_dir}/batch/", exist_ok=True)
    pairs = [(c, p) for c in args.style_code for p in range(len(prompts))]
    pending = [(c, p) for c, p in pairs if not (args.skip_existing and os.path.exists(image_path(c, p)))]
    print(f"{len(pending)} of {len(pairs)} images to generate")

    # The text encoder and the transformer do not fit on the GPU together: encode every
    # (style code, prompt) pair first, then free the text encoder and denoise.
    style_ids = make_style_ids_fn(args, style_generator, code_freq)
    encoded = {}
    for style_code in tqdm(dict.fromkeys(c for c, _ in pending), desc="encoding"):
        generated_ids = style_ids(style_code)
        if args.style_mode == "ordered":
            print(f"style code {style_code} -> section levels {style_ids.levels(style_code)}")
        for prompt_i in [p for c, p in pending if c == style_code]:
            encoded[style_code, prompt_i] = pipeline.encode_inputs(
                image=[placeholder_image],
                prompt=prompts[prompt_i],
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                num_images_per_prompt=1,
                codebook_id=generated_ids,
            )
    del pipeline.text_encoder
    torch.cuda.empty_cache()

    if args.accelerate:
        sample_steps = 4
    else:
        sample_steps = 40

    # Batch codes that share a prompt: their embeddings have the same length, and every forward
    # pass streams the whole transformer from CPU, so a batch costs little more than one image.
    batches = []
    for prompt_i in range(len(prompts)):
        codes = [c for c, p in pending if p == prompt_i]
        batches += [(prompt_i, codes[i:i + args.batch_size]) for i in range(0, len(codes), args.batch_size)]
    for prompt_i, codes in tqdm(batches, desc="generating"):
        set_seed(args.seed)
        batch = [encoded[c, prompt_i] for c in codes]
        inputs = {
            "image": [placeholder_image],
            # One generator per image, all with the same seed: every image starts from the same noise.
            "generator": [torch.Generator().manual_seed(args.seed) for _ in codes],
            "true_cfg_scale": true_cfg_scale,
            "num_inference_steps": sample_steps,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            # in place of prompt / negative_prompt
            **{k: torch.cat([e[k] for e in batch]) for k in batch[0]},
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
        for style_code, image in zip(codes, output.images):
            image.save(image_path(style_code, prompt_i))

    res_imgs = [[Image.open(image_path(c, p)) for p in range(len(prompts))] for c in args.style_code]
    seeds = list(args.style_code)

    concatenate_images_with_sref(res_imgs, seeds).save(f"{output_dir}/batch.png")
    print(f"The results are saved to {output_dir}/batch.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--style_code",
        type=int,
        nargs="+",
        default=[1234567, 5201314, 13415926, 886, 20010627, 996007, 2333],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained_models",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
    )

    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default="./test_prompts.txt",
    )
    parser.add_argument(
        "--freq_threshold",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--freq_k",
        type=float,
        default=0.0001,
    )

    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="codes generated together per forward pass (same prompt)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="do not regenerate images that already exist in --output_path")
    add_style_code_args(parser)

    args = parser.parse_args()
    main(args)
