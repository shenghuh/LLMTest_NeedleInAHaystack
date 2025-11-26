import os

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs)  # make a copy
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key

        # For modern models (gpt-4.1, 4o, 5, 5-mini, etc.) we must **explicitly**
        # set temperature=1, otherwise LangChain will default to 0.7 and the API
        # will reject it.
        if self._uses_modern_api():
            self.model_kwargs["temperature"] = 1
        # For legacy models we keep whatever was passed (often temperature=0).

        self.evaluator = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            **self.model_kwargs,
        )

    def _uses_modern_api(self) -> bool:
        modern_prefixes = (
            "gpt-4.1",
            "gpt-4o",
            "gpt-4.5",
            "gpt-5",
            "gpt-5.1",
            "gpt-5-mini",
        )
        return self.model_name.startswith(modern_prefixes)

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])
