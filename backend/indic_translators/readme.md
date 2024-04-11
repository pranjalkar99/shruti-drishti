Title: IndicTrans2 Inference API with FastAPI

# Introduction:

This repository provides a production-ready inference API built upon the AI4BHARAT IndicTrans2 models (https://github.com/AI4Bharat/IndicTrans2). It enables you to seamlessly translate text between the 22 scheduled Indic languages with high quality.

# Key Features:

* Robust Multilingual Translation: Supports all 22 scheduled Indic languages, including multiple scripts for low-resource languages.
* FastAPI Integration: Leverages FastAPI for a user-friendly and performant API experience.
* Scalability: Designed for scalability to handle diverse translation needs.
* Open Source: Freely available for customization and integration into your projects.

# Installation
Instructions to setup and install everything before running the code.

## Clone the github repository and navigate to the project directory.
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2

## Install all the dependencies and requirements associated with the project.
source install.sh

## Download the models from huggingface
[English to Indic](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
[Indic to English](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)

## Usage

```python
from translation_engine import translate_sentences
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

input_sentences = ['जिनेवा में अंतर्राष्ट्रीय जलवायु सम्मेलन में भाग लेते हुए, संयुक्त राष्ट्र महासचिव ने सतत ऊर्जा समाधान विकसित करने और जीवाश्म ईंधन से दूर जाने के लिए वैश्विक सहयोग की तत्काल आवश्यकता पर जोर दिया, इस बात पर प्रकाश डालते हुए कि इस जटिल चुनौती से निपटने और सभी के लिए एक स्वच्छ भविष्य सुनिश्चित करने के लिए अनुसंधान और ज्ञान-साझाकरण पर अंतर्राष्ट्रीय सहयोग महत्वपूर्ण है।']
ckpt_dir = "indictrans2-en-indic-1B"
translations = translate_sentences(input_sentences,
                                  direction = "indic-en",
                                  src_lang = "hin_Deva",
                                  tgt_lang = "eng_Latn",)
print(translations)
```

These are the available language and language codes:

| Language Codes      |                                  |                                |
|---------------------|----------------------------------|--------------------------------|
| Assamese (asm_Beng) | Kashmiri (Arabic) (kas_Arab)     | Punjabi (pan_Guru)             |
| Bengali (ben_Beng)  | Kashmiri (Devanagari) (kas_Deva) | Sanskrit (san_Deva)            |
| Bodo (brx_Deva)     | Maithili (mai_Deva)              | Santali (sat_Olck)             |
| Dogri (doi_Deva)    | Malayalam (mal_Mlym)             | Sindhi (Arabic) (snd_Arab)     |
| English (eng_Latn)  | Marathi (mar_Deva)               | Sindhi (Devanagari) (snd_Deva) |
| Konkani (gom_Deva)  | Manipuri (Bengali) (mni_Beng)    | Tamil (tam_Taml)               |
| Gujarati (guj_Gujr) | Manipuri (Meitei) (mni_Mtei)     | Telugu (tel_Telu)              |
| Hindi (hin_Deva)    | Nepali (npi_Deva)                | Urdu (urd_Arab)                |
| Kannada (kan_Knda)  | Odia (ory_Orya)                  |                                |
| Kannada (kan_Knda)  | Odia (ory_Orya)                  |                                |
