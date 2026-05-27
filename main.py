import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core import run_llm

# Instantiate system tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger("API_Gateway")

app = FastAPI(
    title="Enterprise RAG Gateway Engine",
    description="Production-grade asynchronous inference pipeline for document retrieval contexts",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Text string query directed to the documentation store",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Factual answer compiled by agent node")
    context: List[Dict[str, Any]] = Field(
        default=[], description="Source records pulled for contextual grounding"
    )


@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_rag_inference(payload: QueryRequest):
    """
    Inference endpoint consuming unstructured search prompts, triggering internal
    database evaluation graphs, and passing back validated answers.
    """
    logger.info(f"Received semantic query transaction: '{payload.query}'")
    result = await run_llm(payload.query)

    if "internal system timeout error" in result["answer"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector retrieval infrastructure failure encountered.",
        )

    return QueryResponse(answer=result["answer"], context=result["context"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
