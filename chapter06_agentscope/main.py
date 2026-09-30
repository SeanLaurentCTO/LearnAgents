"""启动Review版三国狼人杀。

运行方式：

    python -X utf8 -m chapter06_agentscope.main

完整游戏会进行多轮Vertex AI调用，运行时间和Token消耗取决于游戏轮数。
"""

import asyncio

from chapter06_agentscope.agents import create_game_agents
from chapter06_agentscope.game import WerewolfGame
from chapter06_agentscope.game_models import (
    GameRole,
    GameState,
    PlayerState,
)
from chapter06_agentscope.model_client import create_vertex_ai_chat_model


def create_initial_game_state() -> GameState:
    """创建身份固定的初始对局。

    复习阶段使用固定身份，便于复现问题。后续若要随机身份，应由Python
    打乱身份列表，不能让LLM自行决定或修改身份。
    """
    return GameState(
        players=[
            PlayerState(name="曹操", role=GameRole.WEREWOLF),
            PlayerState(name="司马懿", role=GameRole.WEREWOLF),
            PlayerState(name="诸葛亮", role=GameRole.SEER),
            PlayerState(name="华佗", role=GameRole.WITCH),
            PlayerState(name="刘备", role=GameRole.VILLAGER),
            PlayerState(name="关羽", role=GameRole.VILLAGER),
        ],
    )


async def main() -> None:
    """组装模型、Agent和游戏控制器，并运行完整游戏。"""
    # 所有玩家共享同一个模型客户端。创建客户端会读取配置并刷新ADC
    # Token，但此时尚未向Gemini发送推理请求。
    model = create_vertex_ai_chat_model(
        stream=False,
        temperature=0.5,
    )

    try:
        state = create_initial_game_state()

        # 创建Agent时只组合Prompt、Formatter和Memory，不调用模型。
        agents = create_game_agents(state, model)

        game = WerewolfGame(
            state=state,
            agents=agents,
            max_action_retries=2,
            max_days=10,
        )

        # 真正的多轮AI调用从这里进入game.py，由不同游戏阶段触发。
        winner = await game.run()

        print("\n========== 最终结果 ==========")
        print(f"获胜阵营：{winner.value}")

        # 游戏结束后才输出完整身份，避免游戏进行时泄露给Agent。
        for player in state.players:
            print(
                f"{player.name}: "
                f"role={player.role.value}, "
                f"alive={player.alive}, "
                f"eliminated_on_day={player.eliminated_on_day}",
            )

    finally:
        # 多个Agent共享同一客户端，因此必须由最外层只关闭一次。
        await model.client.aio.aclose()
        model.client.close()


if __name__ == "__main__":
    asyncio.run(main())
