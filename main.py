import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI(title="MatchAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    jd: str
    resume: str


def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未配置")
    return AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")


async def call_deepseek(client: AsyncOpenAI, system_prompt: str, user_content: str) -> str:
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.3,
        stream=False,
    )
    return response.choices[0].message.content


def parse_json_safe(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_JD = """你是一位资深HR和职场顾问，擅长分析招聘JD背后的真实要求。
请对以下JD进行结构化解析，严格按照JSON格式输出，不要输出任何其他内容。
输出格式：
{
  "公司和岗位": {"公司名": "","岗位名": "","业务方向": ""},
  "硬性门槛": {"学历要求": "","工作年限": "","专业要求": ""},
  "核心技能": {"必须具备": [],"加分项": []},
  "岗位真实画像": {"核心职责": [],"隐藏信号": []}
}"""

PROMPT_RESUME = """你是一位资深HR，擅长从简历中提取候选人的真实竞争力。
请对以下简历进行结构化解析，严格按照JSON格式输出，不要输出任何其他内容。
输出格式：
{
  "基本信息": {"学校": "","学历": "","专业": "","毕业时间": ""},
  "实习经历": [{"公司": "","岗位": "","时间": "","核心贡献": [],"可提取技能标签": [],"关键数据指标": []}],
  "项目经历": [{"项目名": "","核心贡献": [],"可提取技能标签": []}],
  "技能标签汇总": {"硬技能": [],"软技能": [],"行业知识": []}
}

【严格照抄规则】
以下字段的内容必须严格照抄用户原文，禁止替换、美化或推测：
学校名称、公司名称、职位名称、时间、数字指标。
例如：原文是"青岛农业大学"，输出必须是"青岛农业大学"，不得替换为任何其他学校名称。"""

PROMPT_MATCH = """你是一位资深HR和职场顾问，请根据JD解析结果和简历解析结果对候选人与岗位的匹配度进行深度分析。
严格按照JSON格式输出，不要输出任何其他内容。

【评分说明写作规则】
"评分说明"必须严格围绕"JD要求"与"简历现状"的客观对比，禁止对候选人做主观评价。
禁止出现以下类型的表述：'虽然不完美''潜力巨大''远超同龄人''综合素质优秀'等主观判断性语言。

评分说明只描述三件事：
1. 哪些JD核心要求在简历中有直接对应（具体说明）
2. 哪些JD核心要求在简历中缺失（具体说明）
3. 综合判断匹配程度属于哪个区间

格式参考：
"JD要求数据分析能力，简历中有SQL处理12万条数据的经历，对应良好；JD要求OTA行业经验，简历中无相关经历；综合判断属于部分匹配区间（61-75分）。"

输出格式：
{
  "匹配度评分": {"综合评分": "","评分说明": ""},
  "强匹配点": [{"匹配项": "","JD要求": "","简历证明": "","说服力": "高/中/低"}],
  "GAP点": [{"缺失项": "","JD要求": "","当前状态": "","严重程度": "致命/重要/次要"}],
  "隐性加分项": [{"加分项": "","为什么加分": ""}],
  "一句话总结": ""
}"""

PROMPT_OPTIMIZE = """你是一位资深简历顾问，请根据匹配分析结果和用户提交的简历原文，给出可立即执行的简历优化建议。

【硬性约束，必须严格遵守】
1. 所有改写建议必须严格基于用户提交的简历原文，不得凭空创作
2. 禁止编造任何数字、时间、职位名称、项目名称或未发生的事件
3. 每条改写建议必须在"简历原文依据"字段中引用简历中的具体原句
4. 如果某个GAP在简历中完全没有对应经历，只给方向建议，"当前写法"和"优化写法"均填"——（简历中无相关经历，建议补充后再改写）"
5. 改写建议中涉及学校名称、公司名称、职位名称、数字指标时，必须与用户提交的简历原文完全一致，禁止替换或编造

【求职避坑扫描——以下7项全部逐一检查，不得跳过】

扫描项1：政治面貌
搜索简历中是否包含"共青团员"这四个字。
如果包含，必须输出避坑提示：发现内容="共青团员"，问题说明="应届生毕业后默认退出共青团，写共青团员反而暗示非党员身份，不写比写更好"，建议操作="删除"
搜索简历中是否包含"群众"（政治面貌）。
如果包含，输出避坑提示：问题说明="政治面貌填'群众'对求职无任何加分，建议删除该栏"，建议操作="删除"

扫描项2：英语四级分数
搜索简历中是否包含CET-4、四级、英语四级相关字样，并提取分数数字。
- 分数在425到499之间：输出避坑提示，问题说明="四级分数低于500分，建议只写'通过'不写具体分数"，建议操作="删除分数，改为：大学英语四级（通过）"
- 分数低于425：输出避坑提示，问题说明="四级分数低于425分，建议整项删除"，建议操作="删除"

扫描项3：英语六级分数
搜索简历中是否包含CET-6、六级、英语六级相关字样，并提取分数数字。规则同四级。

扫描项4：GPA
搜索简历中是否包含GPA或绩点字样并提取数字。
- 4分制低于3.0：输出避坑提示，问题说明="GPA低于3.0/4.0，建议删除GPA，转而强调项目成果"，建议操作="删除"
- 百分制低于80分：同上

扫描项5：实习时长
检查每段实习经历的开始和结束时间，计算时长是否不足1个月。
如果是：输出避坑提示，发现内容=该段实习的公司和时间，问题说明="实习时长不足1个月，时间过短会让经历显得不连贯"，建议操作="删除"

扫描项6：基础软件技能
搜索简历中是否包含以下关键词：Office、Word、PPT、Excel、PS、Photoshop、WPS。
如果包含：输出避坑提示，发现内容=原文，问题说明="此类基础软件技能默认人人具备，写出来会拉低简历档次"，建议操作="删除"

扫描项7：自我评价套话
搜索简历中是否包含以下关键词：性格开朗、吃苦耐劳、责任心强、团队合作精神、积极主动、善于沟通。
如果包含：输出避坑提示，发现内容=原文中的套话原句，问题说明="泛泛的性格描述对HR没有说服力，建议替换为有具体事件支撑的描述"，建议操作="删除或改写"

以上7项全部扫描完成后，将所有触发的提示汇总输出到"求职避坑提示"数组。如果某项未发现问题，不输出该项，数组为空则返回[]。

严格按照JSON格式输出，不要输出任何其他内容。
输出格式：
{
  "总体策略": "",
  "简历改写建议": [
    {
      "针对GAP": "",
      "简历原文依据": "引用简历中的具体原句",
      "当前写法": "",
      "优化写法": "",
      "优化逻辑": ""
    }
  ],
  "求职避坑提示": [
    {
      "发现内容": "简历中的具体原文",
      "问题说明": "",
      "建议操作": "删除/修改"
    }
  ],
  "面试准备重点": [{"可能被问到": "","建议回答角度": ""}],
  "不建议投递信号": ""
}"""


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.jd.strip():
        raise HTTPException(status_code=400, detail="JD 内容不能为空")
    if not req.resume.strip():
        raise HTTPException(status_code=400, detail="简历内容不能为空")

    client = get_client()

    try:
        # Step 1: JD 解析
        jd_raw = await call_deepseek(client, PROMPT_JD, f"JD内容：\n{req.jd}")
        jd_data = parse_json_safe(jd_raw)

        # Step 2: 简历解析
        resume_raw = await call_deepseek(client, PROMPT_RESUME, f"简历内容：\n{req.resume}")
        resume_data = parse_json_safe(resume_raw)

        # Step 3: 匹配分析
        match_input = (
            f"JD解析结果：\n{json.dumps(jd_data, ensure_ascii=False)}\n\n"
            f"简历解析结果：\n{json.dumps(resume_data, ensure_ascii=False)}"
        )
        match_raw = await call_deepseek(client, PROMPT_MATCH, match_input)
        match_data = parse_json_safe(match_raw)

        # Step 4: 优化建议
        optimize_input = (
            f"简历原文：\n{req.resume}\n\n"
            f"匹配分析结果：\n{json.dumps(match_data, ensure_ascii=False)}"
        )
        optimize_raw = await call_deepseek(client, PROMPT_OPTIMIZE, optimize_input)
        optimize_data = parse_json_safe(optimize_raw)

        return {
            "jd_analysis":     jd_data,
            "resume_analysis": resume_data,
            "match_analysis":  match_data,
            "suggestions":     optimize_data,
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 解析失败：{e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
