from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GameModel(BaseModel):
    """所有游戏状态和动作模型的共同基类。"""

    model_config = ConfigDict(
        # 拒绝未定义字段，防止LLM自行扩展结构化动作格式。
        extra="forbid",
        # 自动去除字符串首尾空白。
        str_strip_whitespace=True,
        # 对模型创建后的字段重新赋值继续执行Pydantic校验。
        validate_assignment=True,
    )


class GameRole(str, Enum):
    """玩家在狼人杀中的隐藏身份。"""

    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"


class GamePhase(str, Enum):
    """游戏控制器允许进入的阶段。"""

    NIGHT_WEREWOLF = "night_werewolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH = "night_witch"
    DAY_ANNOUNCEMENT = "day_announcement"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    FINISHED = "finished"


class PlayerState(GameModel):
    """单个玩家的权威状态。

    该模型由Python游戏控制器维护，不允许LLM直接修改。
    """

    name: str = Field(
        min_length=1,
        max_length=30,
        description="玩家名称，例如诸葛亮、曹操",
    )

    role: GameRole = Field(
        description="玩家的隐藏身份",
    )

    alive: bool = Field(
        default=True,
        description="玩家当前是否存活",
    )

    eliminated_on_day: int | None = Field(
        default=None,
        ge=1,
        description="玩家在哪一天出局；仍然存活时为None",
    )


class GameCamp(str, Enum):
    """游戏获胜阵营。"""

    GOOD = "good"
    WEREWOLF = "werewolf"


class NightState(GameModel):
    """
    当前夜晚产生的确定性临时状态。
    这些字段由游戏控制器维护，不能由LLM直接修改。
    """

    werewolf_target: str | None = Field(
        default=None,
        description="狼人当晚选择的袭击目标",
    )

    poison_target: str | None = Field(
        default=None,
        description="女巫当晚选择的毒药目标",
    )

    antidote_used: bool = Field(
        default=False,
        description="女巫当晚是否使用了解药",
    )


class GameState(GameModel):
    """一局狼人杀的全局权威状态。"""

    day: int = Field(
        default=1,
        ge=1,
        description="当前游戏天数",
    )

    phase: GamePhase = Field(
        default=GamePhase.NIGHT_WEREWOLF,
        description="当前游戏阶段",
    )

    players: list[PlayerState] = Field(
        min_length=1,
        description="本局游戏中的全部玩家",
    )

    # 女巫的药物是确定性资源，由Python维护。
    antidote_available: bool = Field(
        default=True,
        description="女巫的解药是否仍可使用",
    )

    poison_available: bool = Field(
        default=True,
        description="女巫的毒药是否仍可使用",
    )

    night: NightState = Field(
        default_factory=NightState,
        description="当前夜晚的临时状态",
    )

    winner: GameCamp | None = Field(
        default=None,
        description="获胜阵营；游戏未结束时为None",
    )

    @model_validator(mode="after")
    def validate_unique_player_names(self) -> Self:
        """确保同一局游戏中不存在同名玩家。

        验证器应抛出ValueError，Pydantic会在模型创建边界自动将其
        包装成包含上下文信息的ValidationError。
        """

        names = [player.name for player in self.players]
        if len(names) != len(set(names)):
            raise ValueError("当前游戏中（同一局）存在同名玩家")

        return self

    @property
    def alive_player_names(self) -> list[str]:
        """返回全部存活玩家名称。"""
        return [player.name for player in self.players if player.alive]

    def get_player(self, name: str) -> PlayerState:
        """按名称返回玩家；玩家不存在时抛出ValueError。"""
        for player in self.players:
            if player.name == name:
                return player

        raise ValueError(f"当前玩家不存在：{name}")

class DiscussionResult(GameModel):
    """玩家在白天讨论阶段的结构化发言。"""

    speech: str = Field(
        min_length=1,
        max_length=1000,
        description="玩家公开发表的言论",
    )

    suspected_player: str | None = Field(
        default=None,
        max_length=30,
        description="当前最怀疑的玩家；没有明确对象时为None",
    )
#
class WerewolfAction(GameModel):
    """狼人夜间选择袭击目标的动作。"""

    target: str = Field(
        min_length=1,
        max_length=30,
        description="狼人准备袭击的玩家",
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
        description="选择该目标的理由",
    )

class SeerAction(GameModel):
    """预言家夜间查验玩家的动作。"""

    target: str = Field(
        min_length=1,
        max_length=30,
        description="预言家准备查验的玩家",
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
        description="选择该玩家进行查验的理由",
    )

class WitchAction(GameModel):
    """女巫夜间使用解药或毒药的动作。

    当前游戏规则规定：

    1. 女巫可以不使用任何药；
    2. 使用毒药时必须指定下毒目标；
    3. 不使用毒药时不能指定下毒目标；
    4. 同一夜不能同时使用解药和毒药。
    """

    use_antidote: bool = Field(
        default=False,
        description="是否使用解药救下当晚被袭击的玩家",
    )

    use_poison: bool = Field(
        default=False,
        description="是否使用毒药",
    )

    poison_target: str | None = Field(
        default=None,
        max_length=30,
        description="毒药目标；不使用毒药时必须为None",
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
        description="女巫执行该动作的理由",
    )

    @model_validator(mode="after")
    def validate_medicine_choices(self) -> Self:
        """检查女巫的药物选择是否自洽。"""
        if self.use_poison and self.poison_target is None:
            raise ValueError("使用毒药时必须指定poison_target")

        if not self.use_poison and self.poison_target is not None:
            raise ValueError("不使用毒药时不能指定poison_target")

        if self.use_antidote and self.use_poison:
            raise ValueError("同一夜不能同时使用解药和毒药")

        return self

class VoteAction(GameModel):
    """玩家在白天放逐投票阶段提交的动作。"""

    target: str = Field(
        min_length=1,
        max_length=30,
        description="投票放逐的目标玩家",
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
        description="投票给该玩家的理由",
    )
