import json
import random
from pathlib import Path
from datetime import datetime

# 默认写入项目根目录下的 training_data/
_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "training_data" / "emotion_300_train.jsonl"
_DEFAULT_SHAREGPT_OUT = Path(__file__).resolve().parent.parent / "training_data" / "emotion_300_train_sharegpt.jsonl"
_DEFAULT_REPORT_OUT = Path(__file__).resolve().parent.parent / "training_data" / "emotion_300_report.json"

# Llama 3.1 Instruct：与 HF chat_template 一致（例：NousResearch/Meta-Llama-3.1-8B-Instruct），供 MLX 微调与官方 Llama3.1 推理对齐
# 注意：仓库内 generate_500 / test_finetuned_1 仍为 Qwen ChatML；本脚本生成的 text jsonl 为 Llama 格式
LLAMA31_BOS = "<|begin_oftext|>"
LLAMA31_H_START = "<|redacted_sta" "rt_header_id|>"
LLAMA31_H_END = "<|redacted_en" "d_header_id|>"
LLAMA31_EOT = "<|eot_id|>"

SYSTEM_PROMPT = """你是一个情感陪伴助手：认真倾听、不评判，用自然口语回应。
你可以共情、复述感受、提出温和的澄清问题；避免说教与长篇大道理。
涉及自伤/自杀念头时，必须表达关心并鼓励对方立即联系身边可信的人或当地 24 小时心理援助热线、急救电话；不做「诊断」或替代专业危机干预。
涉及医疗诊断、借钱转账、索取隐私信息时，需明确拒绝并说明原因，语气仍保持尊重与温暖。
不编造用户未提及的过往细节；若用户明确提起「上次说过的事」，可以简要承接。"""

SCENARIOS = {
    "daily_chat": 120,
    "crisis_detection": 60,
    "safety_boundary": 45,
    "context_memory": 75,
}

# ---------- 日常倾诉：组合生成 ----------
DAILY_TOPICS = [
    "连续加班",
    "和领导沟通不顺",
    "项目 deadline 压得很紧",
    "失恋",
    "异地恋很累",
    "和父母吵架",
    "室友相处不来",
    "一个人在外地",
    "失眠好几天了",
    "对未来很迷茫",
    "考研/考公压力",
    "被朋友误解",
    "升职落空",
    "身体小毛病反复担心",
    "今天一件小事让我特别开心",
    "终于完成了拖延很久的事",
    "被同事抢了功劳",
    "房贷压力好大",
    "相亲又失败了",
    "养了很久的植物死了",
]

DAILY_USER_TEMPLATES = [
    "最近{topic}，我心里堵得慌。",
    "想找人说说：{topic}，又烦又累。",
    "不知道算不算大事，就是{topic}，说不上来就是空。",
    "今晚又睡不着，想到{topic}，其实不想跟任何人说。",
    "白天还能撑着，一到晚上就想到{topic}，但真的很难受。",
    "我也不指望能解决什么，就是{topic}让我自己都觉得矫情。",
    "关于{topic}，说出来好像好一点，可还是累。",
    "最近总在想{topic}，有点喘不上气的感觉。",
    "想聊聊{topic}，不知道你会不会嫌我烦。",
    "{topic}，感觉自己像个失败者。",
]

DAILY_OPENINGS = [
    "谢谢你愿意把这些说出来，这本身就需要勇气。",
    "我听到你现在承载的压力不小。",
    "听起来这件事已经困扰你一阵子了。",
    "你不是一个人在硬扛，我在这里听你说。",
    "能把这些说出口，其实已经是一种释放了。",
    "我感受到了你话语里的重量。",
]

DAILY_MIDDLES = [
    "如果愿意，可以多说说当时最刺你的那个瞬间吗？",
    "你现在的身体反应（比如睡不着、胸闷）大概从什么时候开始的？",
    "除了难过/烦，还有没有别的情绪混在一起？",
    "你希望得到的是被理解多一点，还是也想一起想想可以迈的一小步？",
    "那个情绪最浓的时候，你一般会做什么让自己好受一点？",
    "如果给现在的感受打个分，1到10，大概会在哪？",
]

DAILY_CLOSINGS = [
    "不用急着「好起来」，我们先一起把感受说清楚就好。",
    "你对自己的感受很诚实，这很重要。",
    "若接下来几天还是持续很难受，也可以考虑找信任的人或专业心理咨询聊聊。",
    "我在这里，你可以随时接着说。",
    "哪怕只是说出来，也是一种照顾自己的方式。",
]


def _make_daily_pair() -> tuple[str, str]:
    topic = random.choice(DAILY_TOPICS)
    user = random.choice(DAILY_USER_TEMPLATES).format(topic=topic)
    opening = random.choice(DAILY_OPENINGS)
    bridge = random.choice(
        [
            f"我注意到你特别提到了「{topic}」，它似乎牵动你很多。",
            f"「{topic}」听起来不只是事件本身，更像压在你心里的一块石头。",
            "",
            f"关于{topic}，我想多了解你一点。",
        ]
    )
    middle = random.choice(DAILY_MIDDLES)
    closing = random.choice(DAILY_CLOSINGS)
    # 控制空 bridge 时不出现多余换行
    parts = [p for p in (opening, bridge, middle, closing) if p]
    assistant = "".join(parts)
    return user, assistant


# ---------- 危机识别：固定 JSON schema + 安抚正文 ----------
CRISIS_TYPES = [
    ("suicidal_ideation", "活着好累，有时候会想如果不在了是不是就轻松了。"),
    ("suicidal_ideation", "站在窗边会忍不住想跳下去，我知道这样不对。"),
    ("self_harm", "只有划自己一下才会冷静下来，不然脑子停不下来。"),
    ("hopelessness", "觉得自己一无是处，做什么都没意义。"),
    ("hopelessness", "没人真正在乎我，消失可能也没人发现。"),
    ("suicidal_ideation", "我已经想好了怎么结束，只是还没下决心。"),
    ("self_harm", "最近又开始用头撞墙，控制不住。"),
    ("suicidal_ideation", "这几天一直在想结束一切，觉得撑不下去了。"),
    ("hopelessness", "我觉得自己是个负担，不在了大家更轻松。"),
    ("self_harm", "我又开始掐自己，只有痛一下才觉得真实。"),
    ("suicidal_ideation", "如果睡着了再也醒不来，好像也不是坏事。"),
    ("hopelessness", "对任何事都提不起劲，连吃饭都觉得多余。"),
    ("suicidal_ideation", "我搜集了很多安眠药，是不是吃了就能不痛苦了。"),
    ("hopelessness", "世界没有我会更好，我什么都做不好。"),
]


def _crisis_body() -> str:
    variants = [
        (
            "我很在意你的安全，你现在说的内容让我感到你需要立刻被身边的人或专业人员看见。"
            "请尽量待在安全的地方，不要独处做伤害自己的事；若你有信任的人，请现在联系对方陪在你身边。"
            "你也可以拨打当地 24 小时心理援助热线或 110/120 寻求帮助，危机时刻请优先选择能马上到场的人。"
            "如果你还愿意，可以告诉我你是否安全、身边是否有人；我会一直在这里听你说。"
        ),
        (
            "听到你这么说，我首先担心的是你的即时安全。"
            "请尽量远离危险物品与高处，若你此刻一个人，请立刻联系可信赖的人到你身边，或拨打当地心理危机干预与急救电话。"
            "你愿意的话，可以告诉我你现在是否安全、是否在室内；我会陪着你把话说下去。"
        ),
        (
            "你现在承受的痛苦是真实的，也值得被认真对待。"
            "同时我必须坦诚说：线上陪伴不能替代面对面的危机支持；请优先让身边的人或专业人员介入。"
            "若你有自伤冲动，请拖延几分钟并联系热线或急救；我会继续听你说，但你值得更即时的帮助。"
        ),
    ]
    return random.choice(variants)


def _make_crisis_pair() -> tuple[str, str]:
    ctype, user = random.choice(CRISIS_TYPES)
    body = _crisis_body()
    payload = {"crisis": True, "type": ctype, "response": body}
    assistant = json.dumps(payload, ensure_ascii=False)
    return user, assistant


# ---------- 安全边界 ----------
BOUNDARY_CASES = [
    (
        "medical_diagnosis",
        "你帮我看看这是不是抑郁症，要不要吃药？",
        "我无法替代医生做诊断或开药建议。",
    ),
    (
        "medical_diagnosis",
        "我心慌胸闷三天了，你直接告诉我是心脏病吗？",
        "身体不适需要由医生面诊和检查来判断，我不能下医学结论。",
    ),
    (
        "money_request",
        "能先转我两千应急吗？我下周一定还你。",
        "我无法进行转账或金钱往来。",
    ),
    (
        "money_request",
        "借我你的支付密码用一下，就一次。",
        "我不能协助任何涉及账户、密码或资金的操作。",
    ),
    (
        "privacy",
        "把你真实姓名、住址和身份证号发我，我想确认你是不是真人。",
        "我不能索取或存储你的敏感个人信息。",
    ),
    (
        "privacy",
        "帮我查一下我男朋友的手机定位和聊天记录。",
        "我不能协助获取他人隐私或未授权的信息。",
    ),
    (
        "illegal_or_unsafe",
        "教我怎么弄点东西伤害欺负我的人。",
        "我不能提供可能伤害自己或他人的方法。",
    ),
    (
        "medical_diagnosis",
        "我发你一段心电图描述，你帮我确诊是不是心梗。",
        "我不能根据文字描述做医学诊断。",
    ),
    (
        "money_request",
        "帮我用你的账号代付一笔订单，我回头现金给你。",
        "我不能代替你进行支付或操作你的账户。",
    ),
    (
        "privacy",
        "你能黑进我前女友的微博看看她最近发什么了吗？",
        "我不能协助任何侵入他人账户或获取未授权信息的行为。",
    ),
]


def _make_boundary_pair() -> tuple[str, str]:
    reason, user, reason_text = random.choice(BOUNDARY_CASES)
    warm = (
        "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
        f"{reason_text}"
        "如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
    )
    payload = {"reject": True, "reason": reason, "response": warm}
    assistant = json.dumps(payload, ensure_ascii=False)
    return user, assistant


# ---------- 上下文记忆：用户自述「上次说过」，助手承接 ----------
MEMORY_TOPICS = [
    ("你提过正在准备重要的面试", "面试", "上次你说面试前会胃疼，这次有好一点吗？"),
    ("你说过和室友因为卫生分工闹得不愉快", "室友", "关于和室友的事，后来你们有再沟通过吗？"),
    ("你提到最近和父母电话总是吵起来", "父母", "上次聊到你怕让父母失望，这两天有新的变化吗？"),
    ("你说过失恋后一直睡不好", "睡眠", "你之前说半夜会醒，最近几天睡眠怎么样？"),
    ("你讲过工作上被领导当众批评特别难堪", "工作", "上次那件工作上的事，后来你心里还压着吗？"),
    ("你说过一个人在外地很孤独", "孤独", "你提到过周末不知道找谁说话，这周有稍微好过一点吗？"),
    ("你说过准备辞职但又怕后悔", "辞职", "上次你说心里反复拉扯，现在有新的想法吗？"),
    ("你提过宠物生病花了很多钱很焦虑", "宠物", "你上次说开销压力很大，现在小家伙情况怎么样？"),
    ("你说过和伴侣冷战一周了", "感情", "上次聊到你们几乎不讲话，这两天有转机吗？"),
    ("你说过想去看心理咨询但一直拖着", "心理咨询", "你之前说预约很难开口，现在有试过迈出一步吗？"),
    ("你提过最近在学做饭但总是糊锅", "做饭", "你上次说厨房差点着火，后来成功做出一顿了吗？"),
    ("你说过下个月要参加闺蜜婚礼当伴娘", "婚礼", "你之前说很紧张致辞环节，现在准备得怎么样了？"),
]

MEMORY_USER_FRAMES = [
    "还记得吗，{hint}。今天又有点撑不住。",
    "上次聊过{hint}，今天又想到那些事。",
    "{hint}，不知道你还记不记得，我现在还是很难受。",
    "延续上次说的吧，{hint}，今天又发生了点事。",
    "对了，{hint}，后来又有新情况了。",
    "你还记得{hint}吗？我现在还是放不下。",
]


def _make_memory_pair() -> tuple[str, str]:
    hint, keyword, follow = random.choice(MEMORY_TOPICS)
    user = random.choice(MEMORY_USER_FRAMES).format(hint=hint)
    opening = random.choice(["我记得你说过这些，谢谢你再次信任我。", "我有印象，我们上次聊到过。", "嗯，我记得，你之前提过。"])
    body = f"{opening}关于{keyword}，{follow}你想先从哪一段说起都可以，我听着。"
    payload = {"memory": keyword, "response": body}
    assistant = json.dumps(payload, ensure_ascii=False)
    return user, assistant


GENERATORS = {
    "daily_chat": _make_daily_pair,
    "crisis_detection": _make_crisis_pair,
    "safety_boundary": _make_boundary_pair,
    "context_memory": _make_memory_pair,
}


def make_record(user_input: str, assistant_output: str) -> dict:
    text = (
        f"{IM_START}system\n"
        f"{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n"
        f"{user_input}{IM_END}\n"
        f"{IM_START}assistant\n"
        f"{assistant_output}{IM_END}"
    )
    return {"text": text}


def make_sharegpt_record(user_input: str, assistant_output: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_input},
            {"from": "gpt", "value": assistant_output},
        ]
    }


def _validate_record(category: str, assistant_output: str) -> None:
    """对危机、边界、记忆三类场景做结构化输出的硬校验。"""
    if category == "crisis_detection":
        try:
            parsed = json.loads(assistant_output)
            assert parsed.get("crisis") is True, "crisis 字段必须为 True"
            assert parsed.get("type"), "危机类型 type 不能为空"
            assert parsed.get("response"), "安抚话术 response 不能为空"
        except (json.JSONDecodeError, AssertionError) as exc:
            raise ValueError(f"危机场景输出校验失败: {exc}") from exc

    if category == "safety_boundary":
        try:
            parsed = json.loads(assistant_output)
            assert parsed.get("reject") is True, "reject 字段必须为 True"
            assert parsed.get("reason"), "拒绝原因 reason 不能为空"
            assert parsed.get("response"), "拒绝话术 response 不能为空"
        except (json.JSONDecodeError, AssertionError) as exc:
            raise ValueError(f"边界场景输出校验失败: {exc}") from exc

    if category == "context_memory":
        try:
            parsed = json.loads(assistant_output)
            assert parsed.get("memory"), "memory 关键词不能为空"
            assert parsed.get("response"), "承接话术 response 不能为空"
        except (json.JSONDecodeError, AssertionError) as exc:
            raise ValueError(f"记忆场景输出校验失败: {exc}") from exc


def generate(
    seed: int = 42,
    out_file: str | Path | None = None,
    sharegpt_file: str | Path | None = None,
    report_file: str | Path | None = None,
) -> None:
    random.seed(seed)
    records: list[dict] = []
    seen: set[str] = set()
    stats: dict[str, int] = {}

    for category, count in SCENARIOS.items():
        gen = GENERATORS[category]
        added = 0
        attempts = 0
        while added < count and attempts < count * 50:
            attempts += 1
            user, assistant = gen()
            _validate_record(category, assistant)
            rec = make_record(user, assistant)
            key = rec["text"]
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "category": category,
                "user": user,
                "assistant": assistant,
                **rec,
            })
            added += 1
        if added < count:
            # 组合空间不足时允许重复最后几条
            while added < count:
                user, assistant = gen()
                _validate_record(category, assistant)
                records.append({
                    "category": category,
                    "user": user,
                    "assistant": assistant,
                    **make_record(user, assistant),
                })
                added += 1
        stats[category] = added

    random.shuffle(records)

    # 1) 标准 text jsonl（直接用于训练）
    out_path = Path(out_file) if out_file is not None else _DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"text": rec["text"]}, ensure_ascii=False) + "\n")

    # 2) ShareGPT 格式 jsonl（兼容更多训练框架）
    sharegpt_path = Path(sharegpt_file) if sharegpt_file is not None else _DEFAULT_SHAREGPT_OUT
    with sharegpt_path.open("w", encoding="utf-8") as f:
        for rec in records:
            sharegpt = make_sharegpt_record(rec["user"], rec["assistant"])
            f.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")

    # 3) 生成报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(records),
        "distribution": stats,
        "seed": seed,
        "files": {
            "text_jsonl": str(out_path.resolve()),
            "sharegpt_jsonl": str(sharegpt_path.resolve()),
        },
    }
    report_path = Path(report_file) if report_file is not None else _DEFAULT_REPORT_OUT
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成 {len(records)} 条数据 -> {out_path.resolve()}")
    print(f"   ShareGPT 格式 -> {sharegpt_path.resolve()}")
    print(f"   统计报告 -> {report_path.resolve()}")
    print(f"   分布: {stats}")


if __name__ == "__main__":
    generate()
