from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel
import torch
from transformers import Constraint


class BitsAndBytes4bitQuantType(str, Enum):
    fp4 = "fp4"
    nf4 = "nf4"


class BitsAndBytesConfigHelper(BaseModel):
    bnb_4bit_compute_dtype: torch.dtype=torch.float32
    bnb_4bit_quant_type: BitsAndBytes4bitQuantType=BitsAndBytes4bitQuantType.fp4
    bnb_4bit_use_double_quant: bool=False
    llm_int8_enable_fp32_cpu_offload: bool=False
    llm_int8_has_fp16_weight: bool=False
    llm_int8_skip_modules: Optional[list[str]]=None
    llm_int8_threshold: float=6.0
    load_in_4bit: bool=False
    load_in_8bit: bool=False
    #quant_method: str="bitsandbytes"

    class Config:
        arbitrary_types_allowed = True


TokenIds = list[int]

class GenerationConfigHelper(BaseModel):
    max_length: int=20
    max_new_tokens: Optional[int]=None
    min_length: int=0
    min_new_tokens: Optional[int]=None
    early_stopping: bool=False
    max_time: Optional[float]=None

    do_sample: Optional[bool]=False
    num_beams: int=1
    num_beam_groups: int=1
    penalty_alpha: Optional[float]=None
    use_cache: bool=True

    temperature: float=1.0
    top_k: int=50
    top_p: float=1.0
    typical_p: float=1.0
    epsilon_cutoff: float=0.0
    eta_cutoff: float=0.0
    diversity_penalty: float=0.0
    repetition_penalty: float=1.0
    encoder_repetition_penalty: float=1.0
    length_penalty: float=1.0
    no_repeat_ngram_size: int=0
    bad_words_ids: Optional[list[TokenIds]]=None
    force_words_ids: Optional[Union[list[TokenIds], list[list[TokenIds]]]]=None
    renormalize_logits: bool=False
    constraints: Optional[list[Constraint]] = None
    forced_bos_token_id: Optional[int]=None
    forced_eos_token_id: Optional[Union[int, list[int]]]=None
    remove_invalid_values: bool=False
    exponential_decay_length_penalty: Optional[tuple[int, float]]=None
    supress_tokens: Optional[TokenIds]=None
    begin_supress_tokens: Optional[TokenIds]=None
    forced_decoder_ids: Optional[list[TokenIds]]=None
    sequence_bias: Optional[dict[tuple[int,...], float]]=None
    guidance_scale: Optional[float]=None
    low_memory: Optional[bool]=None

    num_return_sequences: int=1
    output_attentions: bool=False
    output_hidden_states: bool=False
    output_scores: bool=False
    return_dict_in_generate: bool=False

    pad_token_id: Optional[int]=None
    bos_token_id: Optional[int]=None
    eos_token_id: Optional[Union[int, TokenIds]]=None

    encoder_no_repeat_ngram_size: int=0
    decoder_start_token_id: Optional[int]=None

    class Config:
        arbitrary_types_allowed = True
