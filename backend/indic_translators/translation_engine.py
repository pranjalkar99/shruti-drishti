import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
DEVICE = 'cuda'
BATCH_SIZE=8
def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def translate_sentences(input_sentences:list, direction:str, ckpt_dir = None, src_lang:str='eng_Latn', tgt_lang:str='hin_Deva', quantization=None):
    """
    Translates a list of input sentences from a source language to a target language using a pre-trained model.
    LIST OF SUPPORTED TRANSLATIONS
        Assamese (asm_Beng) 	Kashmiri (Arabic) (kas_Arab) 	Punjabi (pan_Guru)
        Bengali (ben_Beng) 	Kashmiri (Devanagari) (kas_Deva) 	Sanskrit (san_Deva)
        Bodo (brx_Deva) 	Maithili (mai_Deva) 	Santali (sat_Olck)
        Dogri (doi_Deva) 	Malayalam (mal_Mlym) 	Sindhi (Arabic) (snd_Arab)
        English (eng_Latn) 	Marathi (mar_Deva) 	Sindhi (Devanagari) (snd_Deva)
        Konkani (gom_Deva) 	Manipuri (Bengali) (mni_Beng) 	Tamil (tam_Taml)
        Gujarati (guj_Gujr) 	Manipuri (Meitei) (mni_Mtei) 	Telugu (tel_Telu)
        Hindi (hin_Deva) 	Nepali (npi_Deva) 	Urdu (urd_Arab)
        Kannada (kan_Knda) 	Odia (ory_Orya) 	 


    Args:
        input_sentences (list): A list of sentences to be translated.
        ckpt_dir (str): The directory path where the pre-trained model is stored.
        direction (str): The translation direction, e.g., 'en-indic' for English to Indic languages or 'indic-en' for translation of Indic to english.
        src_lang (str, optional): The source language code. Defaults to 'eng_Latn'.
        tgt_lang (str, optional): The target language code. Defaults to 'hin_Deva'.
        quantization (str, optional): The quantization mode for the model. Defaults to None.

    Returns:
        list: A list of translated sentences.
    """
    if ckpt_dir is None:
        if direction=="en-indic":
            ckpt_dir = "indictrans2-en-indic-1B"
        elif direction=="indic-en":
            ckpt_dir = "indictrans2-indic-en-1B"
        else:
            raise ValueError("Invalid direction. Please specify 'en-indic' or 'indic-en'")

    tokenizer, model = initialize_model_and_tokenizer(ckpt_dir, direction, quantization)
    ip = IndicProcessor(inference=True)

    translations = batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip)
    del model 
    del tokenizer
    torch.cuda.empty_cache()
    return translations


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8

    input_sentences = ['जिनेवा में अंतर्राष्ट्रीय जलवायु सम्मेलन में भाग लेते हुए, संयुक्त राष्ट्र महासचिव ने सतत ऊर्जा समाधान विकसित करने और जीवाश्म ईंधन से दूर जाने के लिए वैश्विक सहयोग की तत्काल आवश्यकता पर जोर दिया, इस बात पर प्रकाश डालते हुए कि इस जटिल चुनौती से निपटने और सभी के लिए एक स्वच्छ भविष्य सुनिश्चित करने के लिए अनुसंधान और ज्ञान-साझाकरण पर अंतर्राष्ट्रीय सहयोग महत्वपूर्ण है।']
    ckpt_dir = "indictrans2-en-indic-1B"
    translations = translate_sentences(input_sentences,
                                       direction = "indic-en",
                                       src_lang = "hin_Deva",
                                       tgt_lang = "eng_Latn",)
    print(translations)