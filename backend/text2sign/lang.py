import os


from typing import List, Literal, TypedDict, Any
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.output_parsers import (
    PydanticOutputParser, 
    OutputFixingParser, 
    RetryOutputParser
    )
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
class BlobClassifier(BaseModel):
    blob : List[Literal['dry', 'parent', 'sign', 'hard', 'screen', 'page', 'exercise', 'healthy', 'doctor', 'bedroom', 'window', 'letter', 'narrow', 'small little', 'university', 'brown', 'ball', 'evening', 'race (ethnicity)', 'colour', 'war', 'beautiful', 'hospital', 'low', 'train', 'book', 'loud', 'priest', 'sunday', 'pink', 'skirt', 'summer', 'clock', 'he', 'sister', 'year', 'adult', 'crowd', 'month', 'student', 'school', 'lamp', 'president', 'long', 'car', 'dirty', 'peace', 'bill', 'suit', 'fan', 'author', 'boat', 'we', 'box', 'rich', 'alright', 'soldier', 'female', 'technology', 'radio', 'afternoon', 'bag', 'tomorrow', 'pant', 'slow', 'warm', 'player', 'energy', 'marriage', 'father', 'black', 'old', 'shallow', 'yesterday', 'patient', 'court', 'secretary', 'cell phone', 'baby', 'computer', 't-shirt', 'cat', 'soft', 'mother', 'medicine', 'green', 'girl', 'good evening', 'monday', 'sad', 'today', 'light', 'cold', 'soap', 'high', 'hot', 'bad', 'heavy', 'pleased', 'tool', 'orange', 'fall', 'happy', 'tight', 'it', 'animal', 'week', 'night', 'hello', 'child', 'religion', 'chair', 'wednesday', 'son', 'blue', 'bird', 'truck', 'ugly', 'paint', 'loose', 'they', 'hour', 'alive', 'neighbour', 'reporter', 'young', 'mean', 'deep', 'paper', 'waiter', 'teacher', 'laptop', 'god', 'sick', 'good morning', 'lock', 'gun', 'bank', 'photograph', 'minute', 'price', 'brother', 'male', 'daughter', 'job', 'deaf', 'door', 'telephone', 'short', 'king', 'ring', 'pencil', 'pen', 'city', 'kitchen', 'bathroom', 'india', 'train ticket', 'bed', 'how are you', 'dog', 'actor', 'fast', 'artist', 'time', 'tuesday', 'weak', 'horse', 'red', 'dress', 'transportation', 'bicycle', 'you', 'strong', 'dream', 'mouse', 'husband', 'thursday', 'winter', 'thick', 'good afternoon', 'park', 'key', 'woman', 'sport', 'police', 'market', 'flat', 'cow', 'street or road', 'restaurant', 'grey', 'dead', 'boy', 'family', 'cheap', 'card', 'white', 'season', 'monsoon', 'team', 'newspaper', 'grandmother', 'queen', 'good night', 'thank you', 'money', 'cool', 'spring', 'office', 'clean', 'hat', 'second', 'science', 'clothing', 'expensive', 'grandfather', 'plane', 'table', 'friday', 'election', 'i', 'television', 'wet', 'friend', 'wife', 'she', 'shirt', 'lawyer', 'famous', 'death', 'blind', 'gift', 'yellow', 'curved', 'train station', 'big large', 'poor', 'pocket', 'ground', 'man', 'temple', 'saturday', 'tall', 'location', 'manager', 'nice', 'house', 'attack', 'library', 'wide', 'store', 'quiet', 'camera', 'bus', 'fish', 'shoes', 'morning', 'thin', 'good', 'new']]
    cot_reason: str = Field(..., description="Step by step breakdown of your reason for the blob output")
    cross_verification : bool = Field(..., description="Have you Double checked your output, and crossverifed if it matches the output schema?")

from zukilangchain import CustomLM
llm = CustomLM(api_key="zu-<ZUKI_API_KEY>", base_url="https://zukijourney.xyzbot.net/v1", model_name="mixtral-8x7b")
from ffmpeg_merge import merge_videos
from langgraph.graph import StateGraph, END

import uuid

prompt = '''<instructions>
You are an expert sign language translator, Based on the user query, your task is to divide the user query into blobs of text.
Do not output anything other than the text blobs in json schema.
1. Only output in the specified format without any preamble or extra information.
2. The categories must be only and only from the given categories.
3. The categories must be in lowercase.
4. Cross verify your results
5. It is okay if the sentence doesnt have some words, you can skip out in your response
6. I will give you $1000 for the correct output and fine you $4000 for the wrong output.
7. Strictly adhere to the standards for formatting, dont output "Here is the given json schema, <json schema>" 
8. Do not hallucinate up categories, use categories from the list available.
9. It is fine if the words do not exactly match. For example: Happlily can be interpreted as happy from the categories.
</instructions>
<examples>
query = "The doctor signed the letter."
Your response: ["doctor", "sign", "letter"]
query = "Hi, How are you doing"
Your response: ["how are you"]
query = "I'll call you tomorrow after work."
Your response: ["cell phone", "tomorrow"]
</examples>
<format instructions>
\n{format_instructions}
</format instructions>
<user query>
\n{query}\n
</user query>
'''

class AgentState(TypedDict):
    query: str
    blob_output: BlobClassifier
    videos: Any

def blob_divider(state):
    query = state['query']
    parser = PydanticOutputParser(pydantic_object=BlobClassifier)
    prompt_temp = PromptTemplate(
        template=prompt,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    completion_chain = prompt_temp | llm | parser
    out = completion_chain.invoke(query)
    state['blob_output'] = out
    return state

def media_parser(state):
    print(state['blob_output'])
    blobs = state['blob_output'].blob
    
    task_id = uuid.uuid4().hex
    vids = merge_videos(task_id, blobs)
    state['videos'] = vids
    return state


workflow = StateGraph(AgentState)
workflow.add_node("blob_divider", blob_divider)
workflow.add_node("media_parser", media_parser)

workflow.set_entry_point("blob_divider")
workflow.add_edge("blob_divider", "media_parser")
workflow.add_edge("media_parser", END)
work = workflow.compile()

if __name__ == '__main__':
    inputs = {"query": "the family had a good evening"}
    response = work.invoke(inputs)
    print(response.)
