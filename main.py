import os
import hydra
from dotenv import load_dotenv
from langsmith import traceable
from omegaconf import DictConfig
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)
from langgraph.graph import StateGraph, START, END
from dataloader.dataloader_factory import DataFactoryRegistry

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8080/v1")
llm = ChatOpenAI(model_name=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, max_tokens=8192)


class DataEnricher(TypedDict):
    example: str
    producer: str
    verifier: str
    corrected: str
    is_verified: bool


class VerifyResult(BaseModel):
    is_good: bool = Field(..., description="Indicates if the enriched data is good or not.")
    feedback: str = Field(..., description="Feedback on the enriched data.")


def create_synonym_problem(state: DataEnricher) -> DataEnricher:
    prompt = [
        SystemMessage(
            content=(
                "You are a helpful assistant that enriches data by creating similar problems provided in the example. "
                "Output ONLY the problem text itself. Do not include any conversational openings (like 'Sure, here is...'), "
                "do not include solutions, and do not include any hints or tips."
            )
        ),
        HumanMessage(
            content=(
                "Create a similar problem to this example: " + state['example'] + "\n"
                "Try changing the wording, the problem setting, etc. "
                "Remember to output strictly the problem statement and nothing else."
            )
        )
    ]
    llm_response = llm.invoke(prompt)
    return {
        "producer": llm_response.content,
        "example": state['example']
    }


def verify_synonym_problem(state: DataEnricher) -> DataEnricher:
    structured_llm = llm.with_structured_output(VerifyResult)
    prompt = [
        SystemMessage(content="You are a helpful assistant that verifies the quality of the enriched data and provides feedback if the quality of the generated problem is not good."),
        HumanMessage(content=f"Original example: {state['example']}\n\nGenerated problem: {state['producer']}\n\nVerify if the generated problem is similar yet not same as the original example and of good quality.")
    ]
    result: VerifyResult = structured_llm.invoke(prompt)
    return {
        "verifier": result.feedback,
        "is_verified": result.is_good
    }


def correct_synonym_problem(state: DataEnricher) -> DataEnricher:
    prompt = [
        SystemMessage(content="You are a helpful assistant that corrects the enriched data based on feedback."),
        HumanMessage(content=f"Original example: {state['example']}\n\nGenerated problem: {state['producer']}\n\nFeedback: {state['verifier']}\n\nCorrect the generated problem based on the feedback.")
    ]
    llm_response = llm.invoke(prompt)
    return {
        "corrected": llm_response.content
    }


def route_verify(state: DataEnricher) -> Literal["end", "correct"]:
    return "end" if state.get("is_verified", False) else "correct"


def build_graph() -> StateGraph:
    graph = StateGraph(DataEnricher)
    graph.add_node("create", create_synonym_problem)
    graph.add_node("verify", verify_synonym_problem)
    graph.add_node("correct_it", correct_synonym_problem)
    graph.add_edge(START, "create")
    graph.add_edge("create", "verify")
    graph.add_conditional_edges("verify", route_verify, {"end": END, "correct": "correct_it"})
    # graph.add_edge("correct_it", "verify")  # This line is commented out to prevent infinite loops in the graph.
    graph.add_edge("correct_it", END)
    return graph


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    obj = DataFactoryRegistry.get_data_loader(cfg.data_type)()
    graph = build_graph().compile()
    for _ in range(cfg.epochs):
        for row in obj.iterate():
            initial_state: DataEnricher = {
                "example": row['data'],
                "producer": "",
                "verifier": "",
                "corrected": "",
                "is_verified": False,
            }
            result = graph.invoke(initial_state)
            dummy_example = result['producer']
            if not result['is_verified']:
                dummy_example = result['corrected']
            obj.impute(row['metadata'], dummy_example)


if __name__ == "__main__":
    main()
