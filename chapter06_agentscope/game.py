"""三国狼人杀游戏控制器。

本模块负责构造消息、控制MsgHub通信范围、调用Agent生成结构化动作、
执行业务校验，并在动作合法后更新Python维护的权威GameState。
"""

from collections import Counter
from collections.abc import Callable
from typing import TypeVar

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.pipeline import MsgHub, fanout_pipeline
from pydantic import BaseModel, ValidationError

from chapter06_agentscope.game_models import (
    DiscussionResult,
    GameCamp,
    GamePhase,
    GameRole,
    GameState,
    NightState,
    PlayerState,
    SeerAction,
    VoteAction,
    WerewolfAction,
    WitchAction,
)
from chapter06_agentscope.game_rules import (
    GameRuleError,
    validate_discussion_result,
    validate_seer_action,
    validate_vote_action,
    validate_werewolf_action,
    validate_witch_action,
)

ActionModel = TypeVar("ActionModel", bound=BaseModel)


class WerewolfGame:
    """三国狼人杀游戏控制器"""

    def __init__(
            self,
            state: GameState,
            agents: dict[str, ReActAgent],
            *,
            max_action_retries: int = 2,
            max_days: int = 10,
    ) -> None:
        """初始化游戏控制器。
        Args:
            state:
                Python维护的权威游戏状态。
            agents:
                玩家名称到ReActAgent的映射。
            max_action_retries:
                LLM返回非法动作时，最多重新请求多少次。
            max_days:
                最大游戏天数，防止测试期间无限循环。
        """
        expected_names = {player.name for player in state.players}
        actual_names = set(agents)

        if expected_names != actual_names:
            raise ValueError(
                "Agent名称必须与GameState玩家名称完全一致："
                f"players={expected_names}, agents={actual_names}",
            )

        self.state: GameState = state
        self.agents: dict[str, ReActAgent] = agents
        self.max_action_retries: int = max_action_retries
        self.max_days: int = max_days

    def _host_message(self, content: str) -> Msg:
        """创建主持人发送的AgentScope消息。"""
        return Msg(
            name="主持人",
            role="user",
            content=content,
        )

    def _alive_players(
            self,
            role: GameRole | None = None,
    ) -> list[PlayerState]:
        """返回存活玩家，可选择按身份过滤。"""
        return [
            player
            for player in self.state.players
            if player.alive and (role is None or player.role == role)
        ]

    def _alive_agents(self) -> list[ReActAgent]:
        """返回全部存活玩家对应的Agent。"""
        return [
            self.agents[player.name]
            for player in self._alive_players()
        ]

    async def _request_action(
            self,
            *,
            agent: ReActAgent,
            prompt: str,
            action_model: type[ActionModel],
            validator: Callable[[ActionModel], None],
    ) -> ActionModel:
        """请求Agent生成结构化动作，并执行业务校验。
        """
        current_prompt = prompt
        last_error: Exception | None = None

        # 首次请求不算“重试”，所以总尝试次数为1 + max_action_retries。
        for attempt in range(1, self.max_action_retries + 2):
            reply = await agent(
                self._host_message(current_prompt),
                structured_model=action_model,
            )

            try:
                action = action_model.model_validate(reply.metadata or {})
                validator(action)
                return action
            except (ValidationError, GameRuleError) as error:
                last_error = error

                if attempt > self.max_action_retries:
                    break

                current_prompt = (
                    "你刚才提交的动作不符合当前游戏规则：\n"
                    f"{error}\n\n"
                    "请根据当前阶段和存活玩家重新选择，并严格返回"
                    "指定的结构化动作。"
                )

        # 只有全部尝试均失败后才抛出异常；不能放在循环内部，否则第一次
        # 非法动作就会终止，max_action_retries将失去作用。
        raise GameRuleError(
            f"玩家{agent.name}多次返回非法动作：{last_error}",
        )


    async def _announce_to_alive(self, content: str) -> None:
        """把主持人消息广播给全部存活玩家。"""
        participants = self._alive_agents()
        if not participants:
            return

        # 这里只广播公告，不要求玩家立即回复，因此关闭自动广播。
        async with MsgHub(
                participants=participants,
                announcement=self._host_message(content),
                enable_auto_broadcast=False,
        ):
            pass

    async def run_werewolf_phase(self) -> None:
        """执行狼人私聊和袭击目标选择。"""
        self.state.phase = GamePhase.NIGHT_WEREWOLF

        wolves = self._alive_players(GameRole.WEREWOLF)
        wolf_agents = [self.agents[player.name] for player in wolves]
        if not wolf_agents:
            return

        valid_targets = [
            player.name
            for player in self._alive_players()
            if player.role != GameRole.WEREWOLF
        ]

        announcement = self._host_message(
            f"第{self.state.day}夜，狼人请睁眼。\n"
            f"可袭击的存活玩家：{valid_targets}\n"
            "请秘密讨论今晚的袭击目标。",
        )

        # 该MsgHub只有狼人参与，因此讨论不会泄露给好人Agent。
        async with MsgHub(
                participants=wolf_agents,
                announcement=announcement,
                name=f"night-{self.state.day}-werewolves",
        ):
            for wolf_agent in wolf_agents:
                await wolf_agent(
                    self._host_message(
                        f"请以{wolf_agent.name}的身份简短提出袭击建议。",
                    ),
                )
        # 当前MVP由第一名存活狼人汇总私聊意见并提交最终结构化动作。
        leader = wolf_agents[0]
        action = await self._request_action(
            agent=leader,
            prompt=(
                "请综合狼人同伴的意见，提交今晚最终袭击目标。\n"
                f"合法目标：{valid_targets}"
            ),
            action_model=WerewolfAction,
            validator=lambda candidate: validate_werewolf_action(
                self.state,
                leader.name,
                candidate,
            ),
        )
        self.state.night.werewolf_target = action.target

    async def run_seer_phase(self) -> None:
        """执行预言家查验，并私下发送阵营结果。"""
        self.state.phase = GamePhase.NIGHT_SEER

        seers = self._alive_players(GameRole.SEER)
        if not seers:
            return

        seer = seers[0]
        seer_agent = self.agents[seer.name]
        valid_targets = [
            player.name
            for player in self._alive_players()
            if player.name != seer.name
        ]

        action = await self._request_action(
            agent=seer_agent,
            prompt=(
                f"第{self.state.day}夜，你可以查验一名存活玩家。\n"
                f"合法目标：{valid_targets}"
            ),
            action_model=SeerAction,
            validator=lambda candidate: validate_seer_action(
                self.state,
                seer.name,
                candidate,
            ),
        )

        target = self.state.get_player(action.target)
        alignment = (
            "狼人"
            if target.role == GameRole.WEREWOLF
            else "好人"
        )
        
        # observe()只把结果写入预言家自己的记忆，不会触发模型调用。
        await seer_agent.observe(
            self._host_message(
                f"私密查验结果：{target.name}属于{alignment}阵营。",
            ),
        )

    async def run_witch_phase(self) -> None:
        """执行女巫用药阶段，并记录药物消耗和毒杀目标。"""
        self.state.phase = GamePhase.NIGHT_WITCH

        witches = self._alive_players(GameRole.WITCH)
        if not witches:
            return

        witch = witches[0]
        witch_agent = self.agents[witch.name]
        attacked_player = self.state.night.werewolf_target

        action = await self._request_action(
            agent=witch_agent,
            prompt=(
                f"第{self.state.day}夜，狼人袭击目标是："
                f"{attacked_player or '无'}。\n"
                f"解药可用：{self.state.antidote_available}\n"
                f"毒药可用：{self.state.poison_available}\n"
                f"当前存活玩家：{self.state.alive_player_names}\n"
                "请选择用药动作，也可以不使用任何药。"
            ),
            action_model=WitchAction,
            # Review版规则直接从GameState读取药物库存，只需额外告诉它
            # 当晚狼人目标是谁。
            validator=lambda candidate: validate_witch_action(
                self.state,
                witch.name,
                candidate,
                attacked_player_name=attacked_player,
            ),
        )

        # Action只是模型提出的候选动作；这里才由Python真正消耗药物，
        # 并把结果写入权威状态。
        if action.use_antidote:
            self.state.antidote_available = False
            self.state.night.antidote_used = True

        if action.use_poison:
            self.state.poison_available = False
            self.state.night.poison_target = action.poison_target

    def eliminate_player(self, player_name: str) -> None:
        """由Python规则引擎执行淘汰，而不是相信LLM宣称的结果。"""
        player = self.state.get_player(player_name)
        if not player.alive:
            return

        player.alive = False
        player.eliminated_on_day = self.state.day

    def resolve_night(self) -> list[str]:
        """结算狼人和女巫动作，返回当晚实际出局名单。"""
        victims: set[str] = set()

        werewolf_target = self.state.night.werewolf_target
        if werewolf_target and not self.state.night.antidote_used:
            victims.add(werewolf_target)

        poison_target = self.state.night.poison_target
        if poison_target:
            victims.add(poison_target)

        for victim in victims:
            self.eliminate_player(victim)

        # 排序使日志和测试结果保持稳定。
        return sorted(victims)

    async def run_night(self) -> list[str]:
        """执行完整夜间流程并结算死亡。"""
        # 只重置上一晚的临时动作，不能恢复已经消耗的女巫药物。
        self.state.night = NightState()

        await self.run_werewolf_phase()
        await self.run_seer_phase()
        await self.run_witch_phase()
        return self.resolve_night()

    async def run_day_announcement(
            self,
            night_victims: list[str],
    ) -> None:
        """向全部存活玩家公布夜间结算结果。"""
        self.state.phase = GamePhase.DAY_ANNOUNCEMENT

        if night_victims:
            content = f"天亮了。昨夜出局的玩家：{'、'.join(night_victims)}。"
        else:
            content = "天亮了。昨夜平安无事。"

        await self._announce_to_alive(content)

    async def run_day_discussion(self) -> None:
        """让全部存活玩家依次生成并公开合法发言。"""
        self.state.phase = GamePhase.DAY_DISCUSSION
        players = self._alive_players()
        participants = [self.agents[player.name] for player in players]

        announcement = self._host_message(
            f"第{self.state.day}天进入公开讨论阶段。\n"
            f"当前存活玩家：{self.state.alive_player_names}",
        )

        # 禁止自动广播：先验证结构化发言，再手动交给其他玩家观察，
        # 避免非法发言在校验前进入所有人的Memory。
        async with MsgHub(
                participants=participants,
                announcement=announcement,
                enable_auto_broadcast=False,
                name=f"day-{self.state.day}-discussion",
        ):
            for player in players:
                agent = self.agents[player.name]
                result = await self._request_action(
                    agent=agent,
                    prompt=(
                        "现在轮到你公开发言。请分析局势，"
                        "可以指出一名怀疑对象。"
                    ),
                    action_model=DiscussionResult,
                    # 用默认参数固定本轮player.name，避免lambda延迟绑定。
                    validator=lambda candidate, name=player.name: (
                        validate_discussion_result(
                            self.state,
                            name,
                            candidate,
                        )
                    ),
                )

                public_reply = Msg(
                    name=player.name,
                    role="assistant",
                    content=result.speech,
                    metadata=result.model_dump(),
                )

                # 发言者自己的输入和回复已由ReActAgent保存；这里只写入
                # 其他玩家Memory，避免发言者记忆中出现重复消息。
                for observer in participants:
                    if observer is not agent:
                        await observer.observe(public_reply)

    async def run_vote(self) -> str | None:
        """并发收集投票；唯一最高票者出局，平票时返回None。"""
        self.state.phase = GamePhase.DAY_VOTE
        players = self._alive_players()
        agents = [self.agents[player.name] for player in players]

        vote_prompt = self._host_message(
            "现在进入放逐投票阶段。\n"
            f"存活玩家：{self.state.alive_player_names}\n"
            "请选择一名其他存活玩家作为投票目标并说明理由。"
            "不能投票给自己。",
        )

        # fanout_pipeline并发调用全部Agent。它收集私密投票回复，
        # 不会像MsgHub那样把每张票自动传播给其他玩家。
        replies = await fanout_pipeline(
            agents,
            vote_prompt,
            enable_gather=True,
            structured_model=VoteAction,
        )

        votes: list[str] = []
        for player, agent, reply in zip(
                players,
                agents,
                replies,
                strict=True,
        ):
            try:
                action = VoteAction.model_validate(reply.metadata or {})
                validate_vote_action(self.state, player.name, action)

            except (ValidationError, GameRuleError) as error:
                # 首轮并发投票不合法时，只让对应玩家单独重试。
                action = await self._request_action(
                    agent=agent,
                    prompt=(
                        f"你刚才的投票不合法：{error}\n"
                        "请重新选择一名合法目标。"
                    ),
                    action_model=VoteAction,
                    validator=lambda candidate, name=player.name: (
                        validate_vote_action(
                            self.state,
                            name,
                            candidate,
                        )
                    ),
                )

            votes.append(action.target)

        vote_counts = Counter(votes)
        highest_votes = max(vote_counts.values())
        candidates = [
            name
            for name, count in vote_counts.items()
            if count == highest_votes
        ]

        # MVP规则：最高票并列时不进行第二轮投票，本轮无人出局。
        if len(candidates) != 1:
            print(f"第{self.state.day}天投票平票，无人出局：{vote_counts}")
            return None

        eliminated = candidates[0]
        self.eliminate_player(eliminated)
        print(
            f"第{self.state.day}天投票结果："
            f"{dict(vote_counts)}，{eliminated}出局。",
        )
        return eliminated

    def check_winner(self) -> bool:
        """按存活阵营人数判断游戏是否结束。"""
        alive_players = self._alive_players()
        wolf_count = sum(
            player.role == GameRole.WEREWOLF
            for player in alive_players
        )
        good_count = len(alive_players) - wolf_count

        if wolf_count == 0:
            self.state.winner = GameCamp.GOOD
            self.state.phase = GamePhase.FINISHED
            return True

        if wolf_count >= good_count:
            self.state.winner = GameCamp.WEREWOLF
            self.state.phase = GamePhase.FINISHED
            return True

        return False

    async def run(self) -> GameCamp:
        """运行昼夜循环，直到某个阵营获胜。"""
        if self.check_winner():
            assert self.state.winner is not None
            return self.state.winner

        while self.state.day <= self.max_days:
            print(f"\n========== 第{self.state.day}夜 ==========")

            night_victims = await self.run_night()
            await self.run_day_announcement(night_victims)

            if self.check_winner():
                break

            print(f"\n========== 第{self.state.day}天 ==========")

            await self.run_day_discussion()
            eliminated = await self.run_vote()

            if eliminated is None:
                await self._announce_to_alive("本轮投票平票，无人出局。")
            else:
                await self._announce_to_alive(
                    f"投票结束，{eliminated}被放逐出局。",
                )

            if self.check_winner():
                break

            self.state.day += 1

        if self.state.winner is None:
            raise RuntimeError(
                f"游戏达到最大天数{self.max_days}，仍未产生获胜阵营",
            )

        winner_text = (
            "好人阵营"
            if self.state.winner == GameCamp.GOOD
            else "狼人阵营"
        )
        await self._announce_to_alive(f"游戏结束，{winner_text}获胜！")
        return self.state.winner
