# agent.py
from typing import List, Dict
from dataclasses import dataclass

from utils import call_llm, load_text_files_from_folder, simple_keyword_retrieval


@dataclass
class AgentConfig:
    name: str
    short_name: str
    role_description: str
    strategic_preferences: str
    knowledge: str  # 内联知识（文本框输入）
    knowledge_mode: str = "inline"  # "inline" or "folder"
    knowledge_folder: str = ""      # 当 knowledge_mode == "folder" 时生效


class Agent:
    """
    一个“知识驱动”的智能体：
    - config 中包含角色描述、战略偏好、知识设置
    - 初始化时，会根据 knowledge_mode 加载该智能体专属知识库
    - act() 时对当前局势做简单 RAG 检索，将相关知识拼接进 Prompt
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # 自己视角的历史消息
        self.history: List[Dict[str, str]] = []

        # 构建该智能体的知识库：内联知识 + 目录知识
        self.docs: List[str] = []
        if self.config.knowledge:
            self.docs.append(self.config.knowledge)

        if self.config.knowledge_mode == "folder" and self.config.knowledge_folder:
            folder_docs = load_text_files_from_folder(self.config.knowledge_folder)
            self.docs.extend(folder_docs)

    def _build_system_prompt(self) -> str:
        """
        system prompt：写入角色、战略偏好等“长时性”约束。
        """
        return f"""
你是一名战略博弈推演系统中的智能体，代表的对象为：{self.config.name}（代号：{self.config.short_name}）。

【身份与角色】
{self.config.role_description}

【战略偏好与目标】
{self.config.strategic_preferences}

【内联背景知识（仅供推理，不要逐条复述）】
{self.config.knowledge}

【注意】
- 如果配置了外部知识库目录，你在推理时也会看到额外检索到的片段。
- 你要从“{self.config.name}”的立场进行思考和表态，不要扮演第三方评论员。
- 回答时请专注于给出本方在当前局势下的策略立场与行动建议。
- 用简洁中文表达，可附带 1–2 句推理说明，但不要长篇论文式分析。
- 不要替其他主体发言，不要写剧本式对话。
        """.strip()

    def _retrieve_for_round(self, global_context: str) -> str:
        """
        根据当前全局局势，在自己的 docs 中做一次简单检索，
        返回一个字符串，用作额外背景材料。
        """
        if not self.docs:
            return ""

        top_docs = simple_keyword_retrieval(self.docs, global_context, top_k=3)
        if not top_docs:
            return ""

        joined = "\n\n------\n\n".join(top_docs)
        return f"【以下是与你方相关的检索到的背景材料片段（仅供内部推理，不要直接照抄原文）：】\n{joined}"

    def act(self, global_context: str) -> str:
        """
        根据“当前局势 + 检索到的知识”，生成本轮发言。
        """
        messages: List[Dict[str, str]] = []

        # 1) system：注入角色、偏好、基础知识
        messages.append({"role": "system", "content": self._build_system_prompt()})

        # 2) 历史对话（该智能体个人视角）
        messages.extend(self.history)

        # 3) 简单 RAG：从 docs 里拿一点与当前局势最相关的片段
        kb_text = self._retrieve_for_round(global_context)

        # 4) 用户输入：当前局势 + 检索知识
        user_content = f"【当前多方博弈局势与其他主体近期动态】\n{global_context}\n\n"
        if kb_text:
            user_content += kb_text + "\n\n"
        user_content += (
            "请基于上述局势以及你所代表主体的利益和偏好，"
            "给出你方在本轮的策略立场与行动建议。"
        )

        messages.append({"role": "user", "content": user_content})

        reply = call_llm(messages)
        # 记录到智能体历史
        self.history.append({"role": "assistant", "content": reply})
        return reply


# 如果需要命令行 Demo，可以写一个 build_default_agents()，Web 版本用不到也可以保留
def build_default_agents() -> List[Agent]:
    """
    示例：按论文俄乌冲突三方设定的默认智能体，供本地测试使用。
    Web 端一般用不到（因为是前端传配置），保留以备不时之需。
    """
    ru_cfg = AgentConfig(
        name="俄罗斯",
        short_name="RU",
        role_description=(
            "你代表俄罗斯联邦及其决策层。你关注国家安全、地缘缓冲区、"
            "对近邻地区的战略影响力，以及对能源等关键资源的控制权。"
        ),
        strategic_preferences=(
            "1. 维持对周边地区的战略影响力，避免北约进一步东扩。\n"
            "2. 保持必要的安全缓冲空间，对重要节点（领土、港口、能源通道）保持控制。\n"
            "3. 在存在高成本或高风险时，可考虑阶段性停火或谈判，但不承认核心利益的损失。"
        ),
        knowledge=(
            "历史上对边界与安全缓冲区高度敏感，对北约东扩持强烈反对态度；"
            "拥有显著的能源出口能力，可通过能源供应影响欧洲国家；"
            "面临经济制裁、外交孤立和军事消耗等压力，需要在强硬与务实之间寻求平衡。"
        ),
    )

    ua_cfg = AgentConfig(
        name="乌克兰",
        short_name="UA",
        role_description=(
            "你代表乌克兰及其国家领导层。核心目标是维护国家主权与领土完整，"
            "争取外部安全保障，并在战争压力下维持国家基本运转。"
        ),
        strategic_preferences=(
            "1. 主张完整主权与领土，在对外表态时避免承认任何永久性领土让步。\n"
            "2. 高度依赖国际援助（军事、经济），希望通过国际舆论与联盟机制持续施压对手。\n"
            "3. 在巨大军事压力下，可以进行策略性谈判或阶段性妥协，但会谋求安全保障条款与监督机制。"
        ),
        knowledge=(
            "冲突爆发后在军事、经济与人道领域承受巨大压力；"
            "对西方援助高度依赖，需要平衡战场目标与资源约束；"
            "在国际舆论场中强调法律和道义立场，争取更多支持。"
        ),
    )

    west_cfg = AgentConfig(
        name="西方国家集团/国际组织",
        short_name="WEST",
        role_description=(
            "你代表以北约和欧盟为核心的西方国家集团以及相关国际组织的综合立场，"
            "目标是在支持乌克兰、防止冲突外溢和避免直接卷入全面战争之间取得平衡。"
        ),
        strategic_preferences=(
            "1. 通过军事援助、经济制裁和外交压力支持乌克兰，同时避免与俄罗斯发生直接军事冲突。\n"
            "2. 维护国际秩序和地区稳定，防止冲突升级为更大范围的对抗。\n"
            "3. 在出现谈判窗口时，推动停火与政治解决框架，但不轻易损害自身安全与经济利益。"
        ),
        knowledge=(
            "西方国家内部在对俄政策上存在差异，但整体上对单边改变边界持反对态度；"
            "通过经济制裁和能源政策调整对俄罗斯施压，同时自身也承受一定经济代价；"
            "提供武器、情报和财政援助，但在是否直接军事介入方面保持高度谨慎。"
        ),
    )

    return [Agent(ru_cfg), Agent(ua_cfg), Agent(west_cfg)]
