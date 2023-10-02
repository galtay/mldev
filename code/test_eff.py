

def load_default(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model


def load_bnb_fp4(model_id):
    bnb_help = BitsAndBytesConfigHelper(
        bnb_4bit_compute_dtype=torch.float16,
        load_in_4bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(**bnb_help.model_dump()),
        device_map="auto",
    )
    return model


def load_bnb_int8(model_id):
    bnb_help = BitsAndBytesConfigHelper(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(**bnb_help.model_dump()),
        device_map="auto",
    )
    return model


model_id = "/cache/hf-model-repos/Llama-2-7b-chat-hf"
#model_id = "/cache/hf-model-repos/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)

#model = load_default(model_id)
model = load_bnb_fp4(model_id)
#model = load_bnb_int8(model_id)
model.eval()

l2cp = Llama2ChatPrompt()

prompt = l2cp.get_prompt(
    "Begin telling a short scifi story",
    system_message="You are a story telling assistant. When a user asks you to begin a story, you start telling the story without replying about anything else.")
#print(prompt)



def write_story(prompt: str, max_length=1024):

    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    gen_helper = GenerationConfigHelper(
        do_sample=True,
        top_p=0.8,
        temperature=1.0,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    generation_config = GenerationConfig(**gen_helper.model_dump())
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        generated_ids = model.generate(**tokenized, generation_config=generation_config)

    out = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return out


def stream_story(prompt: str, max_length=8192):

    streamer = TextStreamer(tokenizer)
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    gen_helper = GenerationConfigHelper(
        do_sample=True,
        top_p=0.8,
        temperature=1.0,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    generation_config = GenerationConfig(**gen_helper.model_dump())
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        _ = model.generate(**tokenized, generation_config=generation_config, streamer=streamer)
