import os
import base64
import json  # Ensure json is imported

import langchain
from langchain.chains import TransformChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain import globals

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Set verbose for debugging
globals.set_debug(True)

# Image loading and encoding function
def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    image_base64 = encode_image(image_path)
    return {"image": image_base64}

# Define the transformation chain to load and encode images
load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)

# Define a Pydantic model for the image information
class ImageInformation(BaseModel):
    """Information about an image."""
    image_description: str = Field(description="a short description of the image")
    people_count: int = Field(description="number of humans on the picture")
    main_objects: list[str] = Field(description="list of the main objects on the picture especially in maritime context")

# Define the image model chain to interact with OpenAI's GPT-4 model
@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=1024)
    # The format of the message is fixed, so we send the data to the model
    msg = model.invoke(
        [HumanMessage(
            content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}}  # Corrected format for image
            ]
        )]
    )
    return msg.content

# Define the output parser for the result
parser = JsonOutputParser(pydantic_object=ImageInformation)

# Define the function to get the image information
def get_image_informations(image_path: str) -> dict:
    vision_prompt = """
    Given the image, provide the following information:
    - A count of how many people are in the image
    - Mention the percentage of Female in the total people count 
    - A list of the main objects present in the image
    - A description of the image with the female especially in maritime context
    """
    vision_chain = load_image_chain | image_model | parser  # Combine the chains
    result = vision_chain.invoke({'image_path': image_path, 'prompt': vision_prompt})

    # Save the result as a JSON file **before** returning it
    with open("image_info.json", "w") as json_file:
        json.dump(result, json_file, indent=4)
    return result


# Call the function and print the result
result = get_image_informations("/Users/jordantaylor/PycharmProjects/gender-modeling/test-set/test_photo.png")
print(result)
