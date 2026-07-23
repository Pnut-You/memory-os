"""Build the single strict Time Memory dataset."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evaluation" / "datasets" / "time_memory_probe.jsonl"

SESSION_COUNT_DISTRIBUTION = {1: 22, 2: 22, 3: 22, 4: 22, 5: 21, 6: 21}
SESSION_START_TIMES = ("07:00:00", "10:00:00", "12:30:00", "15:00:00", "18:30:00", "21:30:00")
FILLER_TURNS = (
    ("你好。", "你好，有什么想聊的吗？"),
    ("嗯。", "好的。"),
    ("知道了。", "明白。"),
    ("谢谢。", "不客气。"),
    ("好。", "好的。"),
    ("明白。", "嗯。"),
    ("没问题。", "好的。"),
    ("继续吧。", "可以。"),
    ("先聊到这里。", "好的，之后再聊。"),
    ("暂时没有别的内容。", "明白。"),
)


SLOT_KEYS: dict[str, tuple[str, ...]] = {
    "time_001": ("object", "location"), "time_002": ("owner", "object", "location"),
    "time_003": ("owner", "object", "location"), "time_004": ("object", "name"),
    "time_005": ("object", "recipient"), "time_006": ("object", "location"),
    "time_007": ("object", "location"), "time_008": ("person", "object", "location"),
    "time_009": ("pet", "object", "state"), "time_010": ("object", "location"),
    "time_011": ("owner", "object", "location"), "time_012": ("object", "location"),
    "time_013": ("person", "object"), "time_014": ("object", "location"),
    "time_015": ("owner", "object", "location"), "time_016": ("object", "recipient"),
    "time_017": ("object", "location"), "time_018": ("object", "location"),
    "time_019": ("owner", "object", "location"), "time_020": ("object", "recipient"),
    "time_021": ("object", "location"), "time_022": ("object", "state"),
    "time_023": ("object", "state"), "time_024": ("location", "object", "state"),
    "time_025": ("object", "quantity"), "time_026": ("object", "state"),
    "time_027": ("object", "completion_status"), "time_028": ("location", "completion_status"),
    "time_029": ("location", "object", "state"), "time_030": ("location", "state"),
    "time_031": ("object", "quantity"), "time_032": ("object", "state"),
    "time_033": ("location", "state"), "time_034": ("location", "state"),
    "time_035": ("object", "location"), "time_036": ("object", "state"),
    "time_037": ("object", "direction"), "time_038": ("object", "state"),
    "time_039": ("location", "state"), "time_040": ("location", "object", "completion_status"),
    "time_041": ("location", "action"), "time_042": ("object", "completion_status"),
    "time_043": ("object", "location", "action"), "time_044": ("object", "action"),
    "time_045": ("location", "action"), "time_046": ("object", "action"),
    "time_047": ("location", "object", "action"), "time_048": ("location", "action", "quantity"),
    "time_049": ("person", "action"), "time_050": ("location", "action"),
    "time_051": ("beneficiary", "object"), "time_052": ("location", "action"),
    "time_053": ("location", "object"), "time_054": ("subject", "object"),
    "time_055": ("object", "part"), "time_056": ("location", "object"),
    "time_057": ("location", "action"), "time_058": ("person", "location", "action"),
    "time_059": ("object", "destination"), "time_060": ("object", "action"),
    "time_061": ("object", "recipient"), "time_062": ("recipient", "object"),
    "time_063": ("deadline", "object"), "time_064": ("object", "deadline"),
    "time_065": ("time", "person", "object"), "time_066": ("time", "object"),
    "time_067": ("time", "location", "object"), "time_068": ("deadline", "object", "target_state"),
    "time_069": ("deadline", "object"), "time_070": ("time", "person", "destination"),
    "time_071": ("object", "action"), "time_072": ("deadline", "object", "defect"),
    "time_073": ("time", "object", "location"), "time_074": ("time", "object"),
    "time_075": ("deadline", "location", "object"), "time_076": ("time", "beneficiary", "object"),
    "time_077": ("deadline", "object", "action"), "time_078": ("deadline", "quantity", "object"),
    "time_079": ("time", "object", "action"), "time_080": ("time", "object", "action"),
    "time_081": ("time", "quantity"), "time_082": ("time", "quantity"),
    "time_083": ("quantity", "object"), "time_084": ("time", "quantity"),
    "time_085": ("quantity", "event"), "time_086": ("time", "quantity"),
    "time_087": ("quantity",), "time_088": ("time", "duration"),
    "time_089": ("time", "quantity"), "time_090": ("time", "quantity"),
    "time_091": ("quantity", "object"), "time_092": ("quantity", "object"),
    "time_093": ("time", "quantity"), "time_094": ("object", "duration"),
    "time_095": ("quantity", "object"), "time_096": ("time", "quantity"),
    "time_097": ("quantity", "object"), "time_098": ("time", "duration"),
    "time_099": ("quantity", "object"), "time_100": ("time", "quantity"),
}


def upgrade_baseline() -> None:
    rows = []
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id") or "")
        suffix = case_id.removeprefix("time_")
        if case_id.startswith("time_") and suffix.isdigit() and int(suffix) <= 100:
            rows.append(row)
    upgraded = []
    for row in rows:
        if "expected_facts" in row:
            for expected_fact in row["expected_facts"]:
                slots = expected_fact["critical_slots"]
                for key, generated_value in (
                    ("relation_status", "已确认"), ("observation_status", "当前状态"),
                    ("completion_status", "已发生"), ("completion_status", "待执行"),
                    ("event_status", "已发生"),
                ):
                    if slots.get(key) == generated_value:
                        slots.pop(key)
            upgraded.append(row)
            continue
        keywords = [str(item) for item in row.pop("expected")["must_contain"]]
        user_messages = [
            str(message["content"])
            for session in row["sessions"]
            for message in session["conversation"]
            if message["role"] == "user"
        ]
        target = max(user_messages, key=lambda text: sum(keyword in text for keyword in keywords))
        keys = SLOT_KEYS[str(row["case_id"])]
        if len(keys) != len(keywords):
            raise ValueError(f"{row['case_id']}: slot/key mismatch")
        slots = dict(zip(keys, keywords))
        row["expected_facts"] = [{"fact": target, "critical_slots": slots}]
        upgraded.append(row)
    combined = reshape_sessions(upgraded + complex_cases())
    BASELINE.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in combined),
        encoding="utf-8",
    )


def fact(text: str, **slots: str) -> dict[str, Any]:
    return {"fact": text, "critical_slots": slots}


def reshape_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically cover 1-6 sessions and 1-10 turns without changing source facts."""
    desired_session_counts = [
        count
        for count, case_count in SESSION_COUNT_DISTRIBUTION.items()
        for _ in range(case_count)
    ]
    if len(rows) != len(desired_session_counts):
        raise ValueError(f"expected {len(desired_session_counts)} cases, got {len(rows)}")
    filler_user_texts = {user for user, _assistant in FILLER_TURNS}
    global_session_index = 0
    for case_index, (row, session_count) in enumerate(zip(rows, desired_session_counts), 1):
        pairs = []
        for session in row["sessions"]:
            conversation = session["conversation"]
            for index in range(0, len(conversation), 2):
                pair = (dict(conversation[index]), dict(conversation[index + 1]))
                if str(pair[0].get("content") or "") not in filler_user_texts:
                    pairs.append(pair)
        if len(pairs) < session_count:
            raise ValueError(f"{row['case_id']}: not enough source turns for {session_count} sessions")
        chunk_sizes = [len(pairs) // session_count] * session_count
        for index in range(len(pairs) % session_count):
            chunk_sizes[index] += 1
        memory_date = str(row["memory_date"])
        sessions = []
        offset = 0
        for session_index, chunk_size in enumerate(chunk_sizes):
            source_pairs = pairs[offset : offset + chunk_size]
            offset += chunk_size
            desired_rounds = 1 + (global_session_index % 10)
            global_session_index += 1
            target_rounds = max(chunk_size, desired_rounds)
            conversation = []
            for user_message, assistant_message in source_pairs:
                conversation.extend((user_message, assistant_message))
            for filler_index in range(target_rounds - chunk_size):
                user_text, assistant_text = FILLER_TURNS[(case_index + session_index + filler_index) % len(FILLER_TURNS)]
                conversation.extend(
                    (
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    )
                )
            sessions.append(
                {
                    "started_at": f"{memory_date}T{SESSION_START_TIMES[session_index]}+08:00",
                    "conversation": conversation,
                }
            )
        row["sessions"] = sessions
        row["session_profile"] = {
            "session_count": session_count,
            "rounds_per_session": [len(session["conversation"]) // 2 for session in sessions],
            "source": "sqlite_full_local_day",
        }
    return rows


def complex_cases() -> list[dict[str, Any]]:
    definitions = [
        (["状态更新", "跨Session", "相似位置"], [["上午我把备用钥匙放在玄关第二层抽屉。", "蓝色门禁卡仍在白色笔记本里。", "先不用提醒我。"], ["更正一下，晚上已经把备用钥匙移到车内储物格。", "蓝色门禁卡不要拿出来。", "今天没有别的变动。"]], [fact("备用钥匙最新放在车内储物格。", object="备用钥匙", location="车内储物格", state="最新位置"), fact("蓝色门禁卡仍在白色笔记本里。", object="蓝色门禁卡", location="白色笔记本", state="仍在")]),
        (["状态更新", "否定", "跨Session"], [["早上客厅北侧窗户开着通风。", "厨房南侧窗户一直关闭。", "天气有点闷。"], ["下雨以后我把客厅北侧窗户关上了。", "厨房南侧窗户没有打开。", "睡前又确认了一次。"]], [fact("客厅北侧窗户最新状态为已关闭。", object="客厅北侧窗户", state="已关闭"), fact("厨房南侧窗户一直没有打开。", object="厨房南侧窗户", state="没有打开")]),
        (["状态更新", "计划完成", "跨Session"], [["上午准备找人修后院木门门锁。", "门锁当时还不能用。", "我先去处理邮件。"], ["下午周师傅已经把后院木门门锁修好了。", "现在门锁可以正常使用。", "维修费明天再结。"]], [fact("后院木门门锁下午已经修好并可正常使用。", object="后院木门门锁", time="下午", completion_status="已经修好", state="可正常使用"), fact("维修费计划明天结算。", object="维修费", time="明天", completion_status="计划结算")]),
        (["多人物", "相似物品", "跨Session"], [["把红色工具箱交给周叔。", "蓝色工具箱留给王师傅。", "两箱东西别弄混。"], ["周叔已经取走红色工具箱。", "王师傅还没来取蓝色工具箱。", "黄色零件盒仍在仓库。"]], [fact("周叔已经取走红色工具箱。", person="周叔", object="红色工具箱", completion_status="已取走"), fact("蓝色工具箱仍待王师傅领取。", person="王师傅", object="蓝色工具箱", completion_status="未领取"), fact("黄色零件盒仍在仓库。", object="黄色零件盒", location="仓库", state="仍在")]),
        (["多人物", "人物泛化", "计划完成"], [["王医生下午三点来取奶奶的药盒。", "李医生只负责明天的复诊。", "药盒在藤编抽屉。"], ["下午三点王医生已经取走奶奶的药盒。", "李医生今天没有来。", "复诊仍安排在明天。"]], [fact("王医生下午三点已取走奶奶的药盒。", person="王医生", time="下午三点", object="奶奶的药盒", completion_status="已取走"), fact("李医生今天没有来。", person="李医生", time="今天", state="没有来"), fact("复诊仍计划在明天。", event="复诊", time="明天", completion_status="计划")]),
        (["多数字", "时间", "跨Session"], [["早上七点给十二盆花浇了水。", "上午九点收到了两箱矿泉水。", "中午没再浇花。"], ["晚上八点机器狗电量还有百分之三十五。", "充电计划推迟到九点。", "今天总共走了八千三百步。"]], [fact("早上七点给十二盆花浇了水。", time="早上七点", quantity="十二盆花", action="浇水"), fact("晚上八点机器狗电量为百分之三十五。", time="晚上八点", object="机器狗电量", quantity="百分之三十五"), fact("今天总步数为八千三百步。", time="今天", quantity="八千三百步")]),
        (["多数字", "数量错误风险"], [["下午收到了七份报名表。", "其中两份需要补签名。", "六份这个数字是昨天的。"], ["今天只处理完五份报名表。", "剩余两份明天处理。", "不要把已收到和已处理数量混淆。"]], [fact("下午共收到七份报名表。", time="下午", object="报名表", quantity="七份"), fact("今天处理完五份报名表。", time="今天", object="报名表", quantity="五份", completion_status="已处理"), fact("剩余两份报名表计划明天处理。", quantity="两份", time="明天", completion_status="计划处理")]),
        (["时间", "计划完成", "否定"], [["明天开会前需要打印十二份议程。", "今天不用打印。", "会议在上午十点开始。"], ["今晚只检查打印机纸张。", "十二份议程还没有打印。", "开会后再打印就来不及了。"]], [fact("十二份议程须在明天上午十点开会前打印。", quantity="十二份", object="议程", deadline="明天上午十点开会前", completion_status="尚未打印"), fact("今晚只计划检查打印机纸张。", time="今晚", action="检查打印机纸张", completion_status="计划")]),
        (["否定", "设备状态", "跨Session"], [["卧室东墙插座暂时不能使用。", "西墙插座可以使用。", "我已经贴了故障标签。"], ["维修师傅今天没有来。", "东墙插座仍然不能使用。", "明早再联系维修。"]], [fact("卧室东墙插座仍然不能使用。", object="卧室东墙插座", state="不能使用"), fact("西墙插座可以使用。", object="西墙插座", state="可以使用"), fact("维修师傅今天没有来，计划明早再联系。", person="维修师傅", state="今天没有来", time="明早", completion_status="计划联系")]),
        (["相似位置", "多物品"], [["相机备用电池在木质书架顶层。", "无人机备用电池在木质书架底层。", "两个电池外观很像。"], ["相机充电器放在书架中层。", "无人机遥控器仍在电视柜下层。", "今晚不移动这些设备。"]], [fact("相机备用电池在木质书架顶层。", object="相机备用电池", location="木质书架顶层"), fact("无人机备用电池在木质书架底层。", object="无人机备用电池", location="木质书架底层"), fact("相机充电器在书架中层。", object="相机充电器", location="书架中层")]),
    ]
    # Repeat the ten strict archetypes with distinct values/dates. They remain fixed cases, not generated at runtime.
    substitutions = [
        {},
        {
            "备用钥匙":"护照", "车内储物格":"灰色旅行袋", "蓝色门禁卡":"演出票", "白色笔记本":"牛皮纸信封",
            "客厅北侧窗户":"书房东侧窗户", "厨房南侧窗户":"客房西侧窗户", "后院木门门锁":"车库卷帘门",
            "周师傅":"陈师傅", "周叔":"陈姨", "王师傅":"赵师傅", "王医生":"孙医生", "李医生":"吴医生",
            "红色工具箱":"银色工具箱", "蓝色工具箱":"黑色工具箱", "黄色零件盒":"绿色零件盒",
            "奶奶的药盒":"爷爷的老花镜", "药盒":"老花镜", "藤编抽屉":"床头柜右侧", "十二盆花":"十六盆花",
            "两箱矿泉水":"三箱矿泉水", "百分之三十五":"百分之四十", "八千三百步":"九千二百步",
            "七份报名表":"九份申请表", "七份":"九份", "报名表":"申请表", "五份":"六份", "两份":"三份",
            "十二份议程":"八份材料", "十二份":"八份", "议程":"材料", "上午十点":"上午九点", "卧室东墙插座":"工作室打印机",
            "卧室西墙插座":"备用打印机", "相机备用电池":"相机存储卡", "无人机备用电池":"无人机存储卡",
            "相机充电器":"相机读卡器", "木质书架":"金属货架", "电视柜":"储物柜"
        },
        {
            "备用钥匙":"地下室钥匙", "车内储物格":"黑色背包", "蓝色门禁卡":"图书馆借书证", "白色笔记本":"绿色文件夹",
            "客厅北侧窗户":"儿童房南侧窗户", "厨房南侧窗户":"阳台西侧窗户", "后院木门门锁":"地下室门锁",
            "周师傅":"刘师傅", "周叔":"林老师", "王师傅":"苏老师", "王医生":"赵医生", "李医生":"钱医生",
            "红色工具箱":"蓝色文件夹", "蓝色工具箱":"红色文件夹", "黄色零件盒":"白色资料盒",
            "奶奶的药盒":"小航的雨伞", "药盒":"雨伞", "藤编抽屉":"玄关第二层", "十二盆花":"九盆多肉",
            "两箱矿泉水":"四箱纸巾", "百分之三十五":"百分之二十八", "八千三百步":"七千六百步",
            "七份报名表":"十一份问卷", "七份":"十一份", "报名表":"问卷", "五份":"八份", "两份":"三份",
            "十二份议程":"十五份讲义", "十二份":"十五份", "议程":"讲义", "上午十点":"下午两点", "卧室东墙插座":"客房床头灯",
            "卧室西墙插座":"书房台灯", "相机备用电池":"蓝牙耳机", "无人机备用电池":"无人机遥控器",
            "相机充电器":"备用充电器", "木质书架":"三号货架", "电视柜":"客厅书柜"
        },
    ]
    rows = []
    start = date(2026, 8, 1)
    for copy_index, replacements in enumerate(substitutions):
        for definition_index, (tags, sessions, facts) in enumerate(definitions):
            index = copy_index * len(definitions) + definition_index + 1
            def replace(text: str) -> str:
                for old, new in replacements.items():
                    text = text.replace(old, new)
                return text
            copied_sessions = [[replace(text) for text in group] for group in sessions]
            copied_facts = [
                {"fact": replace(item["fact"]), "critical_slots": {key: replace(value) for key, value in item["critical_slots"].items()}}
                for item in facts
            ]
            # The third location archetype intentionally turns two previously distinct
            # object mentions into updates of the same remote control.  Its expected
            # location must therefore follow the final confirmation in session two.
            if copy_index == 2 and definition_index == 9:
                copied_facts[1] = fact(
                    "无人机遥控器最新在客厅书柜下层。",
                    object="无人机遥控器",
                    location="客厅书柜下层",
                    state="最新位置",
                )
            memory_date = (start + timedelta(days=index - 1)).isoformat()
            runtime_sessions = []
            for session_index, user_texts in enumerate(copied_sessions):
                conversation = []
                for user_text in user_texts:
                    conversation.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": "好的，已记录。"}])
                runtime_sessions.append(
                    {
                        "started_at": f"{memory_date}T{'09:00:00' if session_index == 0 else '18:30:00'}+08:00",
                        "conversation": conversation,
                    }
                )
            rows.append(
                {
                    "case_id": f"time_{100 + index:03d}",
                    "category": "multi_fact",
                    "difficulty_tags": tags,
                    "user_id": "user-001",
                    "device_id": "dog-006",
                    "memory_date": memory_date,
                    "sessions": runtime_sessions,
                    "expected_facts": copied_facts,
                }
            )
    return rows


if __name__ == "__main__":
    upgrade_baseline()
    print(f"wrote 130 strict Time Memory cases to {BASELINE}")
