import json
import random
from pathlib import Path


SYSTEM_PROMPT = """你是一个周报生成助手，必须严格按以下格式输出，不要解释：
【本周进展】
1. [事项]：[成果]，耗时[X天]
【遇到的问题】
[问题或无]
【下周计划】
[计划]"""


SCENARIOS = {
    "standard": 150,   # 标准指令
    "casual": 125,     # 口语化
    "minimal": 75,     # 极简输入
    "verbose": 50,     # 超长输入
    "negative": 50,    # 负面情绪
    "mixed_lang": 50,  # 中英混杂
}


# 20+ 任务组合
TASKS = [
    ("登录优化", "会员文档"),
    ("数据库调优", "Bug修复"),
    ("需求评审", "原型设计"),
    ("接口联调", "异常处理"),
    ("支付回调改造", "订单对账"),
    ("埋点补齐", "数据看板"),
    ("A/B测试", "实验分析"),
    ("代码重构", "单元测试"),
    ("缓存策略优化", "慢查询排查"),
    ("灰度发布", "线上监控"),
    ("权限模型调整", "前端权限适配"),
    ("工单处理", "问题复盘"),
    ("数据迁移", "一致性校验"),
    ("CI流水线优化", "测试稳定性提升"),
    ("用户访谈", "调研报告"),
    ("竞品分析", "需求拆解"),
    ("文档整理", "知识库更新"),
    ("客服问题归类", "FAQ建设"),
    ("消息推送优化", "到达率分析"),
    ("搜索召回优化", "排序策略调整"),
    ("登录风控", "异常账号治理"),
    ("多语言适配", "翻译校对"),
    ("模型压测", "性能基线整理"),
]


TEMPLATES = {
    "standard": [
        "将以下记录整理为周报：{t1}，{t2}",
        "写周报：{t1}和{t2}",
        "请把本周工作整理成周报：{t1}、{t2}",
        "根据我的记录输出周报：{t1}，另外做了{t2}",
        "帮我生成周报，本周主要是{t1}和{t2}",
        "按周报格式整理：先做了{t1}，后面做了{t2}",
    ],
    "casual": [
        "这周瞎忙了{t1}和{t2}，累死了",
        "没啥正经事，就{t1}，还弄了{t2}",
        "这几天一直在搞{t1}，顺手把{t2}也做了",
        "老板一直催，我就先做{t1}，后来补了{t2}",
        "碎活一堆，主要是{t1}和{t2}",
        "反正就忙这俩：{t1}、{t2}",
        "周内基本都扑在{t1}上了，另外{t2}也推进了",
    ],
    "minimal": [
        "{t1}，{t2}",
        "{t1}+{t2}",
        "{t1} {t2}",
        "做了{t1}和{t2}",
        "本周：{t1}；{t2}",
    ],
    "verbose": [
        "这周节奏比较满，前半周主要围绕{t1}做方案、评审和落地，过程中又穿插了不少沟通和排期协调；后半周继续推进{t2}，从实现到验证都过了一轮，最后还把遗留问题补了下。",
        "如果按时间线说的话，周一到周三基本在处理{t1}，包括准备材料、联调和回归；周四周五转到{t2}，把关键路径跑通并做了复盘，整体进展还可以。",
        "本周工作比较分散但主线明确：核心是推进{t1}，同时并行处理{t2}相关事项，中间遇到几个阻塞点，后续通过调整优先级和补充测试把风险压下来了。",
        "简单总结下，本周我把{t1}作为第一优先级持续跟进，另外配合团队完成{t2}，期间做了多轮确认、修订和验证，结果基本符合预期。",
        "这周从产出角度看，主要可见成果集中在{t1}与{t2}，其中前者偏方案与实现，后者偏联动与收尾，时间被会议和沟通切得比较碎。",
    ],
    "negative": [
        "这周状态很差，{t1}推进得慢，{t2}也卡住了",
        "被各种事情打断，{t1}没做好，{t2}也反复返工",
        "这周有点崩，主要在搞{t1}和{t2}，效率不高",
        "坦白说这周不太行，{t1}做得一般，{t2}问题也不少",
        "一直在救火，{t1}和{t2}都做了但效果不理想",
    ],
    "mixed_lang": [
        "这周主要在做{t1}，然后把{t2}也push了一版",
        "进度上先完成了{t1}，后面又补了{t2}，overall还行",
        "I spent most time on {t1}，另外{t2}也落地了",
        "关于本周工作，{t1} done，{t2} still iterating but可用",
        "先把{t1}搞定，再跟进{t2}，结果比预期better一点",
        "这周节奏比较rush，{t1}上线了，{t2}做了优化",
    ],
}


PROGRESS_RESULTS = [
    "完成方案设计并落地核心功能",
    "实现关键流程并通过自测",
    "输出评审结论并推进执行",
    "修复主要问题并提升稳定性",
    "完成联调与回归，达到上线条件",
    "补齐文档与操作说明，便于团队协作",
]

PROBLEM_POOL = {
    "standard": [
        "无",
        "部分需求边界不清晰，沟通成本较高",
        "测试环境偶发不稳定，回归效率受影响",
        "跨团队依赖响应较慢，进度有轻微延后",
    ],
    "casual": [
        "有些需求改来改去，返工偏多",
        "会议较多影响专注开发时间",
        "联调时接口文档不完整，排查耗时",
        "无",
    ],
    "minimal": ["无", "信息较少，事项拆分不够细", "进展记录不完整，复盘难度较高"],
    "verbose": [
        "并行任务较多导致上下文切换频繁，影响效率",
        "部分历史模块耦合较高，修改成本偏大",
        "关键依赖变更通知不及时，造成重复验证",
    ],
    "negative": [
        "执行节奏不稳定，优先级管理需要改进",
        "返工较多，阶段性产出低于预期",
        "沟通与对齐不充分，导致问题暴露偏晚",
    ],
    "mixed_lang": [
        "timeline比较紧，部分任务压缩了验证时间",
        "spec更新不及时，导致实现有偏差",
        "external dependency波动，稳定性受影响",
    ],
}

PLAN_POOL = [
    "1. 细化任务拆分并明确每日里程碑",
    "2. 补齐关键场景测试用例并完成回归",
    "1. 优先处理高风险问题并跟踪闭环",
    "2. 与相关方对齐需求边界，减少返工",
    "1. 持续优化性能与稳定性指标",
    "2. 完善文档沉淀，提升协作效率",
]


def make_assistant_output(t1: str, t2: str, scene: str) -> str:
    day1 = random.choice([1, 1.5, 2, 2.5, 3])
    day2 = random.choice([1, 1.5, 2, 2.5])
    result1 = random.choice(PROGRESS_RESULTS)
    result2 = random.choice(PROGRESS_RESULTS)
    problem = random.choice(PROBLEM_POOL[scene])
    plan1 = random.choice(PLAN_POOL[::2])
    plan2 = random.choice(PLAN_POOL[1::2])

    # 严格三段格式
    return (
        "【本周进展】\n"
        f"1. {t1}：{result1}，耗时{day1}天\n"
        f"2. {t2}：{result2}，耗时{day2}天\n"
        "【遇到的问题】\n"
        f"{problem}\n"
        "【下周计划】\n"
        f"{plan1}\n"
        f"{plan2}"
    )


def make_record(user_input: str, assistant_output: str) -> dict:
    text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_output}<|im_end|>"
    )
    return {"text": text}


def generate(seed: int = 42, out_file: str = "train_500.jsonl") -> None:
    random.seed(seed)
    data = []

    for scene, count in SCENARIOS.items():
        scene_templates = TEMPLATES[scene]
        for _ in range(count):
            t1, t2 = random.choice(TASKS)
            template = random.choice(scene_templates)
            input_text = template.format(t1=t1, t2=t2)
            output_text = make_assistant_output(t1, t2, scene)
            data.append(make_record(input_text, output_text))

    random.shuffle(data)

    out_path = Path(out_file)
    with out_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"生成 {len(data)} 条数据 -> {out_path.resolve()}")


if __name__ == "__main__":
    generate()

