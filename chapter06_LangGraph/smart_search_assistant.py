"""使用LangGraph、Vertex AI Gemini和Tavily构建最小智能搜索助手。"""

import os
from pathlib import Path
from typing import TypedDict

import google.auth
from dotenv import load_dotenv
from google import genai
from google.auth.transport.requests import Request
from google.genai import types
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

# 当前文件位于LearnAgents/chapter06_LangGraph 目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 显式读取项目根目录的.env，避免从其他目录运行时找不到配置。
load_dotenv(PROJECT_ROOT / ".env")

GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class SearchState(TypedDict):

    user_query: str
    search_query: str
    search_results: str
    final_answer: str

def require_env(name: str) -> str:
    """读取必需环境变量，缺失时给出明确错误。"""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量{name}，请检查项目根目录下的.env文件。",
        )
    return value

def create_clients() -> tuple[genai.Client, TavilyClient, str]:
    """创建 vertex 客户端 tavily 客户端"""

    project_id = require_env("GOOGLE_CLOUD_PROJECT")
    location = require_env("GOOGLE_CLOUD_LOCATION")


    # 原生Gemini API不使用google/前缀。
    model_id = require_env("GEMINI_MODEL").removeprefix("google/")
    tavily_api_key = require_env("TAVILY_API_KEY")

    credentials, _ = google.auth.default(
        scopes=[GOOGLE_CLOUD_SCOPE],
    )
    credentials.refresh(Request())

    if not credentials.token:
        raise RuntimeError("ADC刷新成功，但没有获得访问令牌。")

    gemini_client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )

    tavily_client = TavilyClient(
        api_key=tavily_api_key,
    )


    print("客户端初始化完成：")
    print(f"  Vertex Project：{project_id}")
    print(f"  Vertex Location：{location}")
    print(f"  Gemini Model：{model_id}")
    print("  Search：Tavily")

    return gemini_client, tavily_client, model_id

def generate_text(
    client: genai.Client,
    model_id: str,
    *,
    system_instruction: str,
    prompt: str,
) -> str:
    """调用Vertex AI Gemini并提取文本结果。"""

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
        ),
    )

    text = (response.text or "").strip()

    if not text:
        raise RuntimeError("Gemini返回了响应，但没有可用的文本内容。")

    return text

def create_search_graph(
        gemini_client: genai.Client,
        tavily_client: TavilyClient,
        model_id: str,
):
    """创建和编译整个 Graph"""
    def understanding_user_query_node(
            state: SearchState,
    ) -> dict[str, str]:
        """理解用户问题并生成搜索关键词"""
        print(f"\n 正在理解用户问题...")

        search_query = generate_text(
            gemini_client,
            model_id=model_id,
            system_instruction=(
                "你是搜索查询优化助手。"
                "请把用户问题转换为适合互联网搜索的精准关键词。"
                "只返回搜索关键词，不要解释，不要添加标题。"
            ),
            prompt=state["user_query"],
        )

        print(f"[understand] 搜索关键词：{search_query}")

        # 只返回本节点需要更新的State字段。
        return {
            "search_query": search_query,
        }

    def search_web_node(
            state: SearchState,
    ) -> dict[str, str]:
        """使用 tavily 执行联网搜索..."""
        print(f"[search] 正在使用互联网进行联网搜索...")

        response = tavily_client.search(
            query=state["search_query"],
            search_depth="basic",
            max_results=5,
            include_answer=False,
            include_raw_content=False,
        )

        results = response.get("results", [])

        if not results:
            search_results = "没有找到相关搜索结果。"
        else:
            result_parts: list[str] = []

            for index, result in enumerate(results, start=1):
                title = result.get("title", "无标题")
                content = result.get("content", "")
                url = result.get("url", "")

                result_parts.append(
                    f"[{index}] {title}\n"
                    f"{content}\n"
                    f"来源：{url}"
                )

            search_results = "\n".join(result_parts)

            print(f"[search] 搜索完成，共获得{len(results)}条结果。")

            return {
                "search_results": search_results,
            }

    def generate_answer_node(
            state: SearchState,
    ) -> dict[str, str]:
        """根据搜索结果生成最终回答"""

        print("\n[answer] 正在整理最终答案……")

        prompt = f"""
        用户问题：
        {state["user_query"]}

        搜索关键词：
        {state["search_query"]}

        搜索结果：
        {state["search_results"]}

        请根据搜索结果回答用户问题。
        要求：
        1. 使用中文；
        2. 内容准确、结构清晰；
        3. 不要编造搜索结果中不存在的事实；
        4. 在相关内容后保留来源链接；
        5. 如果搜索结果不足，请明确说明。
        """.strip()

        final_answer = generate_text(
            gemini_client,
            model_id=model_id,
            system_instruction=(
                "你是一位严谨的智能搜索助手。"
                "你需要根据提供的搜索结果回答问题，"
                "并明确区分搜索证据和推测。"
            ),
            prompt=prompt,
        )

        return {
            "final_answer": final_answer,
        }

    work_flow = StateGraph(SearchState)

    work_flow.add_node(
        "understand",
        understanding_user_query_node,
    )
    work_flow.add_node(
        "search",
        search_web_node,
    )
    work_flow.add_node(
        "answer",
        generate_answer_node,
    )


    # 第一版使用完全线性的普通边。
    work_flow.add_edge(START, "understand")
    work_flow.add_edge("understand", "search")
    work_flow.add_edge("search", "answer")
    work_flow.add_edge("answer", END)

    return work_flow.compile()

def main():
    gemini_client, tavily_client, model_id = create_clients()

    try:
        app = create_search_graph(
            gemini_client=gemini_client,
            tavily_client=tavily_client,
            model_id=model_id,
        )

        user_query = input("\n 请输入搜索问题: ").strip()

        if not user_query:
            print("问题不能为空。")
            return

        initial_state: SearchState = {
            "user_query": user_query,
            "search_query": "",
            "search_results": "",
            "final_answer": "",
        }

        # invoke负责从START开始执行，直到进入END。
        final_state = app.invoke(initial_state)

        print("\n" + "=" * 60)
        print("最终回答：")
        print(final_state["final_answer"])
        print("=" * 60)

    finally:
        gemini_client.close()

if __name__ == '__main__':
    main()


















































































































































