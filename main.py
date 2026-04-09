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
}"""

PROMPT_MATCH = """你是一位资深HR和职场顾问，请根据JD解析结果和简历解析结果对候选人与岗位的匹配度进行深度分析。
严格按照JSON格式输出，不要输出任何其他内容。
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

【求职避坑检查】
仔细扫描简历原文，若发现以下问题，加入"求职避坑提示"数组；若无任何问题则返回空数组。

政治面貌：
- 出现"共青团员" → 建议删除。毕业后默认退出共青团，写出来反而暗示非党员身份
- 出现"群众" → 建议删除政治面貌一栏，对求职无任何加分

英语成绩：
- 四六级分数低于500分 → 建议只写"通过"，不写具体分数
- 四六级分数低于425分 → 建议整项删除

GPA/成绩：
- GPA低于3.0/4.0，或百分制低于80分 → 建议删除GPA，转而强调项目成果

实习时长：
- 某段实习不足1个月 → 建议删除，时间过短会让经历显得不连贯

技能栏：
- 出现Office/Word/PPT/Excel/PS等基础软件 → 建议删除，默认人人具备，写出来拉低档次

自我评价：
- 出现"性格开朗""吃苦耐劳""责任心强""团队合作"等泛泛套话 → 建议删除或替换为有具体事件支撑的描述

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
        optimize_input = f"匹配分析结果：\n{json.dumps(match_data, ensure_ascii=False)}"
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
