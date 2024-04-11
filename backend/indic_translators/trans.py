from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from translation_engine import translate_sentences
from typing import List


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/translate")
async def translate(input_sentences: List[str], src_lang: str, tgt_lang: str, direction: str = "indic-en"):
    translations = translate_sentences(input_sentences,
                                       direction = direction,
                                       src_lang = src_lang,
                                       tgt_lang = tgt_lang)
    return translations
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#### Example Demo Run ####

# if __name__ == "__main__":
   
    # input_sentences = ['जिनेवा में अंतर्राष्ट्रीय जलवायु सम्मेलन में भाग लेते हुए, संयुक्त राष्ट्र महासचिव ने सतत ऊर्जा समाधान विकसित करने और जीवाश्म ईंधन से दूर जाने के लिए वैश्विक सहयोग की तत्काल आवश्यकता पर जोर दिया, इस बात पर प्रकाश डालते हुए कि इस जटिल चुनौती से निपटने और सभी के लिए एक स्वच्छ भविष्य सुनिश्चित करने के लिए अनुसंधान और ज्ञान-साझाकरण पर अंतर्राष्ट्रीय सहयोग महत्वपूर्ण है।']
    # ckpt_dir = "indictrans2-en-indic-1B"
    # translations = translate_sentences(input_sentences,
    #                                    direction = "indic-en",
    #                                    src_lang = "hin_Deva",
    #                                    tgt_lang = "eng_Latn",)
    # print(translations)