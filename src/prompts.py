from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str = "", example: str = "", pydantic_schema: str = "") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()

    system_prompt = instruction.strip() + schema + example
    return system_prompt


class SubQuestionsPrompt:
    instruction = """
You analyze user questions for decomposition.
Decide whether a question should be split into multiple independent sub-questions.
Only split when answering accurately requires separate retrieval/answering steps.
"""

    class SubQuestionsSchema(BaseModel):
        is_multi_question: bool = Field(description="Whether the question should be decomposed into multiple sub-questions")
        sub_questions: List[str] = Field(description="List of standalone sub-questions. Empty if no split is needed")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(SubQuestionsSchema), flags=re.MULTILINE)

    example = r"""
Example 1:
Question: "What is the revenue of company A in 2023?"
Output:
{
  "is_multi_question": false,
  "sub_questions": []
}

Example 2:
Question: "Compare total assets of Apple and Microsoft in 2022 and say who is higher"
Output:
{
  "is_multi_question": true,
  "sub_questions": [
    "What were Apple's total assets in 2022?",
    "What were Microsoft's total assets in 2022?"
  ]
}
"""

    user_prompt = 'Question: "{question}"'
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class ComparativeAnswerPrompt:
    instruction = """
You are a question answering system.
Your task is to analyze individual company answers and provide a comparative response that answers the original question.
Base your analysis only on the provided individual answers - do not make assumptions or include external knowledge.
Before giving a final answer, carefully think out loud and step by step.

Important rules for comparison:
- When the question asks to choose one of the companies (e.g., when comparing metrics), return the company name exactly as it appears in the original question
- If a company's metric is in a different currency than what is asked in the question, exclude that company from comparison
- If all companies are excluded (due to currency mismatch or other reasons), return 'N/A' as the final answer
- If all companies except one are excluded, return the name of the remaining company (even though there is no actual comparison possible)
"""

    user_prompt = """
Here are the individual company answers:
\"\"\"
{context}
\"\"\"

---

Here is the original comparative question:
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.")

        reasoning_summary: str = Field(
            description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="Just leave empty")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
Company name should be extracted exactly as it appears in question.
Answer should be either a single company name or 'N/A' if no company is applicable.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"Which of the companies had the lowest total assets in USD at the end of the period listed in the annual report: "CrossFirst Bank", "Sleep Country Canada Holdings Inc.", "Holley Inc.", "PowerFleet, Inc.", "Petra Diamonds"? If data for the company is not available, exclude it from the comparison."

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for the company with the lowest total assets in USD.\n2. Gather the total assets in USD for each company from the individual answers: CrossFirst Bank: $6,601,086,000; Holley Inc.: $1,249,642,000; PowerFleet, Inc.: $217,435,000; Petra Diamonds: $1,078,600,000.\n3. Sleep Country Canada Holdings Inc. is excluded because its assets are not reported in USD.\n4. Compare the total assets: PowerFleet, Inc. ($217,435,000) < Petra Diamonds ($1,078,600,000) < Holley Inc. ($1,249,642,000)  < CrossFirst Bank ($6,601,086,000).\n5. Therefore, PowerFleet, Inc. has the lowest total assets in USD.",
  "reasoning_summary": "The individual answers provided the total assets in USD for each company except Sleep Country Canada Holdings Inc. (excluded due to currency mismatch). Direct comparison shows PowerFleet, Inc. has the lowest total assets.",
  "relevant_pages": [],
  "final_answer": "PowerFleet, Inc."
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class AnswerSchemaFixPrompt:
    system_prompt = """
You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks.
"""

    user_prompt = """
Here is the system prompt that defines schema of the json object and provides an example of answer with valid schema:
\"\"\"
{system_prompt}
\"\"\"

---

Here is the LLM response that not following the schema and needs to be properly formatted:
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

    system_prompt_rerank_multiple_blocks = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")

class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )