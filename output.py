from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Kwaipilot/KAT-Dev",
    task="conversational"
) 

inputs=input('AI :-  ')
model=ChatHuggingFace(llm=llm)

class Review(TypedDict):
    Key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str,"A brief summary of the review"]
    sentiment: Annotated[str,"Return the sentiment positive or negative"]   

structured_model=model.with_structured_output(Review)    
result=structured_model.invoke(inputs)
print(result)