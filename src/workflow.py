# src/workflow.py

import os
from pathlib import Path
from typing import Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from src.rag import load_and_split_code, create_retriever
from src.models import get_llm, get_embeddings
from src.rag import build_retriever
from src.configls import load_config

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    keys: Dict[str, Any]

def create_rag_chain(retriever, llm):
    """
    Creates a RAG chain for code summarization.
    """
    template = """You are a senior software engineer. Use the following retrieved context to summarize the codebase. 
    The summary should be in a structured format, including:
    1.  **Overall Architecture**: A brief description of the main components and their interactions.
    2.  **Key Modules**: A list of important modules and their functionalities.
    3.  **Potential Issues**: Any obvious anti-patterns, code smells, or areas for improvement you notice.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Graph Nodes ---

def summarize_code(state: GraphState) -> GraphState:
    """
    Summarizes the codebase using the RAG chain.
    """
    print("---SUMMARIZING CODE---")
    state_dict = state['keys']
    retriever = state_dict["retriever"]
    llm = state_dict["llm_summarizer"]
    
    rag_chain = create_rag_chain(retriever, llm)
    summary = rag_chain.invoke("Summarize the codebase.")
    
    state_dict["summary"] = summary
    print("---CODE SUMMARY---")
    print(summary)
    return {"keys": state_dict}

def suggest_improvements(state: GraphState) -> GraphState:
    """
    Analyzes the summary and suggests improvements.
    """
    print("---SUGGESTING IMPROVEMENTS---")
    state_dict = state['keys']
    summary = state_dict["summary"]
    llm = state_dict["llm_advisor"]

    template = """You are a principal software architect. Based on the following code summary, provide a list of actionable suggestions for improvement. 
    Focus on:
    1.  **Code Quality**: Refactoring opportunities, adherence to best practices.
    2.  **Performance**: Potential bottlenecks and optimizations.
    3.  **Security**: Possible vulnerabilities.
    4.  **Maintainability**: Suggestions for better documentation, modularization, etc.

    Code Summary:
    {summary}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    improvement_chain = prompt | llm | StrOutputParser()
    suggestions = improvement_chain.invoke({"summary": summary})
    
    state_dict["suggestions"] = suggestions
    print("---IMPROVEMENT SUGGESTIONS---")
    print(suggestions)
    return {"keys": state_dict}

def run_code_advisor_workflow(
    input_dir: str,
    ignore_patterns: list,
    output_dir: str = "./output",
    model_summarizer: str = "gpt-3.5-turbo",
    model_advisor: str = "gpt-4"
):
    """
    Configures and runs the LangGraph workflow for code analysis.
    """
    print(f"\n🚀 开始工作流分析...")
    print(f"📁 输入目录: {os.path.abspath(input_dir)}")
    
    # Load configuration
    config = load_config()
    config["ignore_dir"] = ignore_patterns
    
    # Get base URLs from config
    base_url1 = config.get("base_url1")
    base_url2 = config.get("base_url2")
    base_url_embed = config.get("base_url_embed")
    
    print("🔍 正在加载并索引代码...")
    # build_retriever now uses the specific embedding config
    retriever = build_retriever(input_dir, config)
    
    print(f"🤖 初始化模型: {model_summarizer} (总结), {model_advisor} (建议)")
    llm_summarizer = get_llm(model_summarizer, api_key_env="OPENAI_API_KEY1", base_url=base_url1, temperature=0.2)
    llm_advisor = get_llm(model_advisor, api_key_env="OPENAI_API_KEY2", base_url=base_url2, temperature=0.4)

    # Define the graph
    workflow = StateGraph(GraphState)
    workflow.add_node("summarize", summarize_code)
    workflow.add_node("suggest", suggest_improvements)

    # Set the entrypoint and edges
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "suggest")
    workflow.add_edge("suggest", END)

    # Compile and run the graph
    app = workflow.compile()
    initial_state = {
        "keys": {
            "retriever": retriever,
            "llm_summarizer": llm_summarizer,
            "llm_advisor": llm_advisor,
        }
    }
    
    print("🕸️  正在运行分析工作流 (这可能需要几分钟)...")
    # Run the graph
    result = app.invoke(initial_state)
    print("✅ 工作流执行完成!")
    
    # Save the result to the output directory
    final_state = result['keys']
    summary = final_state.get("summary", "")
    suggestions = final_state.get("suggestions", "")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 正在保存结果到 {out_path.absolute()}...")
    with open(out_path / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 代码分析摘要\n\n" + summary)
    
    with open(out_path / "suggestions.md", "w", encoding="utf-8") as f:
        f.write("# 代码改进建议\n\n" + suggestions)
    
    print(f"\n✨ [分析完成] 结果已保存至: {out_path.absolute()}")
