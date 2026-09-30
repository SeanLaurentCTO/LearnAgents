import os

import google.auth
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType
from camel.utils import print_text_animated
from colorama import Fore
from dotenv import load_dotenv
from google.auth.transport.requests import Request


load_dotenv()

project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
model_id = os.environ["GEMINI_MODEL"]

# 通过ADC取得Vertex AI短期访问令牌。
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
credentials.refresh(Request())

if not credentials.token:
    raise RuntimeError("ADC 刷新成功，但没有取得访问令牌...")

# Vertex AI的global和区域端点域名不同。
host = (
    "aiplatform.googleapis.com"
    if location == "global"
    else f"{location}-aiplatform.googleapis.com"
)

base_url = (
    f"https://{host}/v1/"
    f"projects/{project_id}/"
    f"locations/{location}/endpoints/openapi"
)

# Vertex OpenAI-compatible endpoint要求google/<model>格式。
vertex_model_id = (
    model_id
    if model_id.startswith("google/")
    else f"google/{model_id}"
)

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type=vertex_model_id,
    url=base_url,
    api_key=credentials.token,
    model_config_dict={
        "temperature": 0.4,
        "max_tokens": 2048
    },
)

task_prompt = """
创作一本关于“拖延症心理学”的短篇科普电子书，目标读者是对心理学感兴趣的普通大众。

协作要求：
1. 心理学家负责提供科学概念、实证依据和干预建议；
2. 作家负责规划写作步骤、提出具体写作请求并关注可读性；
3. 内容应通俗准确，不虚构研究结论；
4. 包含引言、核心章节、案例、改善建议和总结；
5. 每轮只处理一个清晰的写作任务；
6. 完成全部任务后，由作家回复CAMEL_TASK_DONE。
""".strip()

role_play_session = RolePlaying(
    assistant_role_name="心理学家",
    user_role_name="作家",
    task_prompt=task_prompt,
    model=model,
    with_task_specify=False,
    output_language="Chinese"
)

print(Fore.YELLOW + f"协作任务：\n{task_prompt}\n")

chat_turn_limit = 10
n = 0

input_msg = role_play_session.init_chat()

while n < chat_turn_limit:
    n += 1
    assistant_response, user_response = role_play_session.step(input_msg)

    if assistant_response.msg is None or user_response.msg is None:
        print(Fore.RED + "对话提前终止：本轮没有获得完整消息。")
        break

    print_text_animated(
        Fore.BLUE
        + f"作家（AI User）：\n\n{user_response.msg.content}\n",
    )
    print_text_animated(
        Fore.GREEN
        + f"心理学家（AI Assistant）：\n\n"
        f"{assistant_response.msg.content}\n",
    )

    if assistant_response.terminated or user_response.terminated:
        print(Fore.MAGENTA + "CAMEL会话已终止。")
        break
    if "CAMEL_TASK_DONE" in user_response.msg.content:
        print(Fore.MAGENTA + "电子书协作任务已完成。")
        break

    input_msg = assistant_response.msg
print(Fore.YELLOW + f"总共进行了 {n} 轮协作对话。")
















































































































