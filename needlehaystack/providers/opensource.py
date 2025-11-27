import os
from operator import itemgetter
from typing import Optional

from langchain_openai import ChatOpenAI  
from langchain_core.prompts import PromptTemplate
import tiktoken
from transformers import pipeline
import torch

from .model import ModelProvider
from langchain_classic.evaluation import load_evaluator
from ..evaluators import Evaluator


class OpenSource(ModelProvider, Evaluator):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(
        max_new_tokens = 300,
        temperature = 0,
    )

    def evaluate_response(self, input, ref, response):
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.model,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=ref,

            # The question asked
            input=input,
        )

        return int(eval_result['score'])

    def __init__(self,
                 model_name: str = "gpt-oss-20b",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 **kwargs):
        """
        Initializes the OpenAI model provider with a specific model.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo-0125'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """

        self.model_name = model_name.split('/')[-1]
        self.model_kwargs = model_kwargs
        self.model = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
    
    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        response = self.model(
            prompt,
            **self.model_kwargs
        )
        return response[0]['generated_text']
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the OpenAI model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """
        return None
        # template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        # \n ------- \n 
        # {context} 
        # \n ------- \n
        # Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""
        
        # prompt = PromptTemplate(
        #     template=template,
        #     input_variables=["context", "question"],
        # )
        # # Create a LangChain runnable
        # model = ChatOpenAI(temperature=0, model=self.model_name)
        # chain = ( {"context": lambda x: context,
        #           "question": itemgetter("question")} 
        #         | prompt 
        #         | model 
        #         )
        # return chain
    

