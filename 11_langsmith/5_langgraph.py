# pip install -U langgraph langchain-openai pydantic python-dotenv langsmith

import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# ---------- Setup ----------
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# ---------- Sample essay ----------
essay2 = """The world is changing rapidly because of a new technology called Artificial Intelligence (AI). Pakistan also has the opportunity to become an important player in this field. If the country invests in education, technology, and innovation, AI can contribute to economic growth and improve people's lives.

Pakistan has several strengths that can support AI development. The country has a young population, talented software engineers, and a growing IT industry. Many startups and technology companies are beginning to adopt AI solutions. The government and educational institutions are also promoting digital skills and encouraging research in emerging technologies.

AI can benefit many sectors in Pakistan. In agriculture, it can help farmers predict weather patterns, detect crop diseases, and improve productivity. In healthcare, AI can assist doctors in diagnosing diseases early and managing patient records more efficiently. In education, AI-powered learning tools can provide personalized education and help students develop modern skills. Government departments can also use AI to improve public services, reduce delays, and increase transparency.

However, Pakistan also faces significant challenges. Many rural areas still lack reliable internet access and digital infrastructure, making it difficult for everyone to benefit from AI. Another concern is that automation may replace certain jobs, requiring workers to learn new skills to remain competitive.

Data privacy and cybersecurity are also important issues. AI systems rely on large amounts of data, so strong laws and regulations are necessary to protect people's personal information and ensure that AI is used responsibly.

To make the most of AI, the government, universities, businesses, and citizens must work together. Investment in education, research, and digital infrastructure is essential. Pakistan should also collaborate with other countries to learn from their experiences and adopt international best practices.

If Pakistan uses AI wisely, it can strengthen its economy, create new employment opportunities, improve public services, and raise the quality of life for its people. However, if AI benefits only a small section of society while others are left behind, it could increase inequality.

In conclusion, the age of AI presents both opportunities and challenges for Pakistan. With careful planning, responsible policies, and equal access to technology, AI can become a powerful tool for national development and help build a brighter future for all Pakistanis."""

# ---------- LangGraph state ----------
class CSSState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]  # merges parallel lists
    avg_score: float

# ---------- Traced node functions ----------
@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: CSSState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: CSSState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: CSSState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: CSSState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}

# ---------- Build graph ----------
graph = StateGraph(CSSState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------- Direct invoke without wrapper ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay",  # becomes root run name
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay2),
                "model": "gpt-4o-mini",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))
