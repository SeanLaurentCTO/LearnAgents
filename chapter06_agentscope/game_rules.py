"""三国狼人杀的确定性业务规则。

``game_models.py``检查字段类型和动作自身的一致性；本模块进一步结合
当前``GameState``判断动作在此时此刻是否允许执行。

这里的函数只做校验，不修改游戏状态，也不调用LLM。真正的消息通信、
动作执行和状态更新将在``game.py``控制器中完成。
"""

from chapter06_agentscope.game_models import (
    DiscussionResult,
    GamePhase,
    GameRole,
    GameState,
    PlayerState,
    SeerAction,
    VoteAction,
    WerewolfAction,
    WitchAction,
)


class GameRuleError(ValueError):
    """动作违反当前游戏业务规则时抛出的异常。"""


def _get_player(state: GameState, player_name: str) -> PlayerState:
    """读取玩家，并把底层查找错误转换成统一的业务规则错误。"""
    try:
        return state.get_player(player_name)
    except ValueError as error:
        raise GameRuleError(f"玩家不在本局游戏中：{player_name}") from error


def _validate_phase(state: GameState, expected_phase: GamePhase) -> None:
    """检查动作是否发生在正确的游戏阶段。"""
    if state.phase != expected_phase:
        raise GameRuleError(
            f"当前阶段是{state.phase.value}，"
            f"该动作只能在{expected_phase.value}阶段执行",
        )


def _validate_actor(
    state: GameState,
    actor_name: str,
    *,
    expected_phase: GamePhase,
    expected_role: GameRole | None = None,
) -> PlayerState:
    """校验动作发起者的阶段、存活状态和身份。

    ``actor_name``必须由Python控制器提供，不能相信LLM自行声明的身份。
    """
    _validate_phase(state, expected_phase)
    actor = _get_player(state, actor_name)

    if not actor.alive:
        raise GameRuleError(f"已出局玩家不能执行动作：{actor_name}")

    if expected_role is not None and actor.role != expected_role:
        raise GameRuleError(
            f"玩家{actor_name}的身份是{actor.role.value}，"
            f"不能执行{expected_role.value}专属动作",
        )

    return actor


def _validate_living_target(
    state: GameState,
    target_name: str,
    *,
    actor_name: str | None = None,
    allow_self: bool = False,
) -> PlayerState:
    """校验目标玩家存在、存活，并按规则判断是否允许选择自己。"""
    target = _get_player(state, target_name)

    if not target.alive:
        raise GameRuleError(f"不能选择已经出局的玩家：{target_name}")

    if not allow_self and actor_name == target_name:
        raise GameRuleError(f"玩家不能选择自己作为目标：{target_name}")

    return target


def validate_discussion_result(
    state: GameState,
    speaker_name: str,
    result: DiscussionResult,
) -> None:
    """校验白天发言者及其可选的怀疑对象。"""
    _validate_actor(
        state,
        speaker_name,
        expected_phase=GamePhase.DAY_DISCUSSION,
    )

    if result.suspected_player is not None:
        _validate_living_target(
            state,
            result.suspected_player,
            actor_name=speaker_name,
        )


def validate_werewolf_action(
    state: GameState,
    actor_name: str,
    action: WerewolfAction,
) -> None:
    """校验狼人夜间袭击动作。

    当前规则规定狼人不能袭击自己，也不能袭击狼人同伴。
    """
    _validate_actor(
        state,
        actor_name,
        expected_phase=GamePhase.NIGHT_WEREWOLF,
        expected_role=GameRole.WEREWOLF,
    )
    target = _validate_living_target(
        state,
        action.target,
        actor_name=actor_name,
    )

    if target.role == GameRole.WEREWOLF:
        raise GameRuleError("狼人不能选择另一名狼人作为袭击目标")


def validate_seer_action(
    state: GameState,
    actor_name: str,
    action: SeerAction,
) -> None:
    """校验预言家夜间查验动作。"""
    _validate_actor(
        state,
        actor_name,
        expected_phase=GamePhase.NIGHT_SEER,
        expected_role=GameRole.SEER,
    )
    _validate_living_target(
        state,
        action.target,
        actor_name=actor_name,
    )


def validate_witch_action(
    state: GameState,
    actor_name: str,
    action: WitchAction,
    *,
    attacked_player_name: str | None,
) -> None:
    """校验女巫夜间用药动作。

    药物库存直接从权威``GameState``读取，而不是相信Prompt或LLM输出。
    该函数只判断是否合法，不会真正消耗解药或毒药。
    """
    _validate_actor(
        state,
        actor_name,
        expected_phase=GamePhase.NIGHT_WITCH,
        expected_role=GameRole.WITCH,
    )

    if action.use_antidote:
        if not state.antidote_available:
            raise GameRuleError("女巫的解药已经使用，不能再次使用")

        if attacked_player_name is None:
            raise GameRuleError("当晚没有狼人袭击目标，不能使用解药")

        # 夜间尚未结算，所以被袭击玩家此时仍应处于存活状态。
        _validate_living_target(
            state,
            attacked_player_name,
            allow_self=True,
        )

    if action.use_poison:
        if not state.poison_available:
            raise GameRuleError("女巫的毒药已经使用，不能再次使用")

        # WitchAction的模型验证通常已保证目标存在；业务层再次防御，
        # 避免未来代码绕过正常的Pydantic构造过程。
        if action.poison_target is None:
            raise GameRuleError("使用毒药时必须指定下毒目标")

        _validate_living_target(
            state,
            action.poison_target,
            actor_name=actor_name,
        )


def validate_vote_action(
    state: GameState,
    voter_name: str,
    action: VoteAction,
) -> None:
    """校验白天放逐投票。

    只有存活玩家可以投票，目标必须存在且存活，并且不能投给自己。
    """
    _validate_actor(
        state,
        voter_name,
        expected_phase=GamePhase.DAY_VOTE,
    )
    _validate_living_target(
        state,
        action.target,
        actor_name=voter_name,
    )
