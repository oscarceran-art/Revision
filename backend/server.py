from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any
import uuid
from datetime import datetime, timezone
from io import BytesIO

from emergentintegrations.llm.chat import LlmChat, UserMessage
from anthropic import AsyncAnthropic
from pypdf import PdfReader
from docx import Document as DocxDocument


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------- MODELS ----------
class Subject(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = ""
    notes: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SubjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    notes: Optional[str] = ""


class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None


class ChatSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New chat"
    subject_id: Optional[str] = None
    personas: List[str] = Field(default_factory=list)
    mode: Literal["solo", "group", "feynman"] = "solo"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New chat"
    subject_id: Optional[str] = None
    personas: List[str] = Field(default_factory=list)
    mode: Literal["solo", "group", "feynman"] = "solo"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    persona_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SendUserMessageRequest(BaseModel):
    session_id: str
    message: str


class StreamReplyRequest(BaseModel):
    session_id: str
    persona_id: Optional[str] = None  # None = default tutor


class ChatSendRequest(BaseModel):
    session_id: str
    message: str


class WorksheetRequest(BaseModel):
    subject_id: Optional[str] = None
    topic: str
    num_questions: int = 5
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "medium"
    question_type: Literal["multiple_choice", "short_answer", "long_answer", "mixed"] = "mixed"
    extra_instructions: Optional[str] = ""


class WorksheetQuestion(BaseModel):
    number: int
    type: str
    question: str
    options: Optional[List[str]] = None
    answer: str
    explanation: Optional[str] = ""
    marks: int = 1


class MarkingFeedback(BaseModel):
    number: int
    awarded: float
    out_of: int
    feedback: str


class MarkingResult(BaseModel):
    total_awarded: float
    total_out_of: int
    percentage: float
    overall_feedback: str
    per_question: List[MarkingFeedback]
    marked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Worksheet(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: Optional[str] = None
    subject_name: Optional[str] = ""
    topic: str
    difficulty: str
    question_type: str
    num_questions: int
    title: str
    instructions: str = ""
    total_marks: int = 0
    duration_minutes: int = 0
    questions: List[WorksheetQuestion]
    user_answers: Dict[str, str] = Field(default_factory=dict)
    marking_result: Optional[MarkingResult] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarkRequest(BaseModel):
    answers: Dict[str, str]  # {question_number: student_answer}


# ---------- HELPERS ----------
def serialize_doc(doc: dict) -> dict:
    """Serialize datetime to ISO string for MongoDB storage."""
    out = dict(doc)
    for k, v in out.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    return out


def parse_datetime(value):
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return datetime.now(timezone.utc)
    return value


async def get_subject(subject_id: str) -> Optional[dict]:
    return await db.subjects.find_one({"id": subject_id}, {"_id": 0})


def build_system_message(subject: Optional[dict]) -> str:
    base = (
        "You are a patient, encouraging revision tutor for a single student. "
        "Explain concepts clearly with short paragraphs, simple examples, and use "
        "headings/bullets when helpful. Ask the student questions occasionally to "
        "check understanding. Keep answers focused and practical."
    )
    if subject:
        ctx = f"\n\nThe current revision subject is: {subject['name']}."
        if subject.get('description'):
            ctx += f"\nSubject description: {subject['description']}"
        if subject.get('notes'):
            notes = subject['notes'][:8000]
            ctx += f"\n\nReference notes provided by the student (use these as ground truth when relevant):\n---\n{notes}\n---"
        base += ctx
    return base


def extract_text_from_upload(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith('.pdf'):
        reader = PdfReader(BytesIO(raw))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts).strip()
    if name.endswith('.docx'):
        doc = DocxDocument(BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    # txt / md / fallback
    try:
        return raw.decode('utf-8', errors='ignore').strip()
    except Exception:
        return ""


# ---------- BASIC ----------
@api_router.get("/")
async def root():
    return {"message": "Revisia API", "model": CLAUDE_MODEL}


@api_router.get("/health")
async def health():
    return {"ok": True, "has_key": bool(ANTHROPIC_API_KEY)}


# ---------- SUBJECTS ----------
@api_router.get("/subjects", response_model=List[Subject])
async def list_subjects():
    docs = await db.subjects.find({}, {"_id": 0}).sort("created_at", -1).to_list(500)
    for d in docs:
        d['created_at'] = parse_datetime(d.get('created_at'))
    return docs


@api_router.post("/subjects", response_model=Subject)
async def create_subject(payload: SubjectCreate):
    obj = Subject(**payload.model_dump())
    await db.subjects.insert_one(serialize_doc(obj.model_dump()))
    return obj


@api_router.get("/subjects/{subject_id}", response_model=Subject)
async def get_subject_endpoint(subject_id: str):
    doc = await get_subject(subject_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Subject not found")
    doc['created_at'] = parse_datetime(doc.get('created_at'))
    return doc


@api_router.patch("/subjects/{subject_id}", response_model=Subject)
async def update_subject(subject_id: str, payload: SubjectUpdate):
    update = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not update:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await db.subjects.update_one({"id": subject_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Subject not found")
    doc = await get_subject(subject_id)
    doc['created_at'] = parse_datetime(doc.get('created_at'))
    return doc


@api_router.delete("/subjects/{subject_id}")
async def delete_subject(subject_id: str):
    result = await db.subjects.delete_one({"id": subject_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Subject not found")
    # Also clean up sessions/messages tied to it (optional)
    await db.chat_sessions.update_many({"subject_id": subject_id}, {"$set": {"subject_id": None}})
    return {"ok": True}


@api_router.post("/subjects/{subject_id}/upload")
async def upload_subject_notes(subject_id: str, file: UploadFile = File(...), append: bool = Form(True)):
    doc = await get_subject(subject_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Subject not found")
    raw = await file.read()
    text = extract_text_from_upload(file.filename or "file.txt", raw)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file")
    new_notes = (doc.get('notes') or '')
    if append and new_notes:
        new_notes = new_notes + "\n\n--- " + (file.filename or 'file') + " ---\n" + text
    else:
        new_notes = text
    await db.subjects.update_one({"id": subject_id}, {"$set": {"notes": new_notes}})
    return {"ok": True, "filename": file.filename, "characters": len(text)}


# ---------- CHAT ----------
@api_router.get("/chat/sessions", response_model=List[ChatSession])
async def list_sessions():
    docs = await db.chat_sessions.find({}, {"_id": 0}).sort("created_at", -1).to_list(500)
    for d in docs:
        d['created_at'] = parse_datetime(d.get('created_at'))
    return docs


@api_router.post("/chat/sessions", response_model=ChatSession)
async def create_session(payload: ChatSessionCreate):
    obj = ChatSession(**payload.model_dump())
    await db.chat_sessions.insert_one(serialize_doc(obj.model_dump()))
    return obj


@api_router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    await db.chat_sessions.delete_one({"id": session_id})
    await db.chat_messages.delete_many({"session_id": session_id})
    return {"ok": True}


@api_router.get("/chat/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_messages(session_id: str):
    docs = await db.chat_messages.find({"session_id": session_id}, {"_id": 0}).sort("created_at", 1).to_list(2000)
    for d in docs:
        d['created_at'] = parse_datetime(d.get('created_at'))
    return docs


@api_router.post("/chat/send", response_model=ChatMessage)
async def send_message(payload: ChatSendRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")

    session = await db.chat_sessions.find_one({"id": payload.session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    subject = None
    if session.get('subject_id'):
        subject = await get_subject(session['subject_id'])

    # Save user message
    user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.message)
    await db.chat_messages.insert_one(serialize_doc(user_msg.model_dump()))

    # Auto-title first message
    msg_count = await db.chat_messages.count_documents({"session_id": payload.session_id})
    if msg_count == 1 and (session.get('title') in (None, '', 'New chat')):
        title = payload.message.strip()[:60]
        await db.chat_sessions.update_one({"id": payload.session_id}, {"$set": {"title": title}})

    system_message = build_system_message(subject)
    # Load full message history for proper multi-turn context
    history_docs = await db.chat_messages.find(
        {"session_id": payload.session_id}, {"_id": 0}
    ).sort("created_at", 1).to_list(2000)
    messages = [{"role": m['role'], "content": m['content']} for m in history_docs]

    try:
        resp = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=system_message,
            messages=messages,
        )
        response_text = resp.content[0].text if resp.content else ""
    except Exception as e:
        logger.exception("Claude error")
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

    ai_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=response_text)
    await db.chat_messages.insert_one(serialize_doc(ai_msg.model_dump()))
    return ai_msg


# ---------- WORKSHEETS ----------
WORKSHEET_SYSTEM = (
    "You are an expert exam-style worksheet generator. You MUST return ONLY valid JSON "
    "matching the requested schema. No prose, no markdown, no code fences."
)


def build_worksheet_prompt(req: WorksheetRequest, subject: Optional[dict]) -> str:
    type_map = {
        "multiple_choice": "Every question must be multiple choice with exactly 4 options labelled A-D.",
        "short_answer": "Every question must be a short-answer question (1-3 sentences).",
        "long_answer": "Every question must be a long-answer / extended response question.",
        "mixed": "Mix of multiple choice (with 4 options), short answer, and one or two long answer questions. Roughly 50% MCQ, 30% short, 20% long.",
    }
    parts = [
        f"Generate a revision worksheet on the topic: \"{req.topic}\".",
        f"Number of questions: {req.num_questions}.",
        f"Difficulty: {req.difficulty}.",
        f"Question style: {type_map[req.question_type]}",
    ]
    if subject:
        parts.append(f"Subject: {subject['name']}.")
        if subject.get('description'):
            parts.append(f"Subject context: {subject['description']}")
        if subject.get('notes'):
            parts.append(f"Use these student notes as ground truth where relevant:\n---\n{subject['notes'][:6000]}\n---")
    if req.extra_instructions:
        parts.append(f"Additional instructions: {req.extra_instructions}")
    parts.append(
        "Return ONLY a JSON object with this exact shape (proper exam paper structure):\n"
        "{\n"
        '  "title": "<exam-style title, e.g. \'Biology Paper 1: Cell Biology\'>",\n'
        '  "instructions": "<2-4 short bullet-style sentences for the front page: what to do, equipment allowed, etc.>",\n'
        '  "duration_minutes": <integer estimated minutes>,\n'
        '  "total_marks": <integer total across all questions>,\n'
        '  "questions": [\n'
        '    {\n'
        '      "number": 1,\n'
        '      "type": "multiple_choice" | "short_answer" | "long_answer",\n'
        '      "question": "<the question>",\n'
        '      "options": ["A) ...", "B) ...", "C) ...", "D) ..."]  (only for multiple_choice, else omit or null),\n'
        '      "marks": <integer marks for this question: MCQ=1, short_answer=2-4, long_answer=5-10>,\n'
        '      "answer": "<the correct/model answer; for MCQ use the letter and full text; for long answer include key points expected>",\n'
        '      "explanation": "<1-3 sentence markscheme notes on what earns marks>"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Total_marks MUST equal the sum of all question marks. Do not wrap in code fences. Do not include any text outside the JSON."
    )
    return "\n\n".join(parts)


def parse_worksheet_json(text: str) -> dict:
    # Strip code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Extract first {...} block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


@api_router.post("/worksheets/generate", response_model=Worksheet)
async def generate_worksheet(req: WorksheetRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")

    subject = None
    if req.subject_id:
        subject = await get_subject(req.subject_id)

    prompt = build_worksheet_prompt(req, subject)
    session_id = f"worksheet-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=ANTHROPIC_API_KEY,
        session_id=session_id,
        system_message=WORKSHEET_SYSTEM,
    ).with_model("anthropic", CLAUDE_MODEL)

    try:
        raw = await chat.send_message(UserMessage(text=prompt))
    except Exception as e:
        logger.exception("Claude error")
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

    try:
        data = parse_worksheet_json(raw)
    except Exception as e:
        logger.error(f"Worksheet parse failed. Raw output: {raw[:1000]}")
        raise HTTPException(status_code=502, detail=f"Could not parse worksheet JSON: {e}")

    questions = []
    for i, q in enumerate(data.get('questions', []), start=1):
        questions.append(WorksheetQuestion(
            number=q.get('number') or i,
            type=q.get('type', 'short_answer'),
            question=q.get('question', ''),
            options=q.get('options') if q.get('options') else None,
            answer=q.get('answer', ''),
            explanation=q.get('explanation', '') or '',
            marks=int(q.get('marks') or (1 if q.get('type') == 'multiple_choice' else 3)),
        ))

    total_marks = data.get('total_marks') or sum(q.marks for q in questions)
    duration = data.get('duration_minutes') or max(10, total_marks * 1)
    instructions = data.get('instructions') or (
        "Answer ALL questions in the spaces provided. "
        "Read each question carefully. "
        "Show your working where appropriate."
    )

    ws = Worksheet(
        subject_id=req.subject_id,
        subject_name=subject['name'] if subject else "",
        topic=req.topic,
        difficulty=req.difficulty,
        question_type=req.question_type,
        num_questions=req.num_questions,
        title=data.get('title') or f"Worksheet: {req.topic}",
        instructions=instructions,
        total_marks=int(total_marks),
        duration_minutes=int(duration),
        questions=questions,
    )
    await db.worksheets.insert_one(serialize_doc(ws.model_dump()))
    return ws


@api_router.get("/worksheets", response_model=List[Worksheet])
async def list_worksheets():
    docs = await db.worksheets.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
    for d in docs:
        d['created_at'] = parse_datetime(d.get('created_at'))
        if d.get('marking_result') and isinstance(d['marking_result'].get('marked_at'), str):
            d['marking_result']['marked_at'] = parse_datetime(d['marking_result']['marked_at'])
    return docs


@api_router.get("/worksheets/{worksheet_id}", response_model=Worksheet)
async def get_worksheet(worksheet_id: str):
    doc = await db.worksheets.find_one({"id": worksheet_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Worksheet not found")
    doc['created_at'] = parse_datetime(doc.get('created_at'))
    if doc.get('marking_result') and isinstance(doc['marking_result'].get('marked_at'), str):
        doc['marking_result']['marked_at'] = parse_datetime(doc['marking_result']['marked_at'])
    return doc


MARKER_SYSTEM = (
    "You are a fair, encouraging exam marker. You MUST return ONLY valid JSON, no prose, no code fences."
)


@api_router.post("/worksheets/{worksheet_id}/mark", response_model=Worksheet)
async def mark_worksheet(worksheet_id: str, payload: MarkRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")
    doc = await db.worksheets.find_one({"id": worksheet_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Worksheet not found")

    # Build marking prompt
    lines = [
        f"Mark this {doc.get('subject_name') or 'revision'} worksheet titled \"{doc['title']}\".",
        "For each question, compare the student's answer to the model answer and award marks fairly.",
        "Give partial credit. Be concise and encouraging in feedback (1-2 sentences each).",
        "",
        "Questions and student answers:",
    ]
    for q in doc['questions']:
        num = q['number']
        student = payload.answers.get(str(num), "").strip() or "[no answer]"
        lines.append(f"\nQ{num} [{q.get('marks', 1)} marks] ({q['type']}): {q['question']}")
        if q.get('options'):
            lines.append("Options: " + " | ".join(q['options']))
        lines.append(f"Model answer: {q['answer']}")
        if q.get('explanation'):
            lines.append(f"Markscheme notes: {q['explanation']}")
        lines.append(f"Student answer: {student}")

    lines.append(
        "\nReturn ONLY this JSON:\n"
        "{\n"
        '  "per_question": [\n'
        '    {"number": <int>, "awarded": <number, can be decimal>, "out_of": <int>, "feedback": "<1-2 sentences>"}\n'
        '  ],\n'
        '  "overall_feedback": "<2-3 sentences summarising strengths and what to revise next>"\n'
        "}"
    )
    prompt = "\n".join(lines)

    chat = LlmChat(
        api_key=ANTHROPIC_API_KEY,
        session_id=f"mark-{uuid.uuid4()}",
        system_message=MARKER_SYSTEM,
    ).with_model("anthropic", CLAUDE_MODEL)

    try:
        raw = await chat.send_message(UserMessage(text=prompt))
    except Exception as e:
        logger.exception("Marker error")
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

    try:
        data = parse_worksheet_json(raw)
    except Exception as e:
        logger.error(f"Marking parse failed: {raw[:500]}")
        raise HTTPException(status_code=502, detail=f"Could not parse marking result: {e}")

    per_q = []
    total_awarded = 0.0
    total_out_of = 0
    by_num = {q['number']: q for q in doc['questions']}
    for item in data.get('per_question', []):
        num = int(item.get('number', 0))
        q = by_num.get(num)
        out_of = int(item.get('out_of') or (q.get('marks', 1) if q else 1))
        awarded = float(item.get('awarded') or 0)
        awarded = max(0.0, min(awarded, out_of))
        per_q.append(MarkingFeedback(
            number=num, awarded=awarded, out_of=out_of,
            feedback=item.get('feedback', '') or ''
        ))
        total_awarded += awarded
        total_out_of += out_of

    if total_out_of == 0:
        total_out_of = sum(q.get('marks', 1) for q in doc['questions'])

    result = MarkingResult(
        total_awarded=round(total_awarded, 1),
        total_out_of=total_out_of,
        percentage=round((total_awarded / total_out_of) * 100, 1) if total_out_of else 0.0,
        overall_feedback=data.get('overall_feedback', '') or '',
        per_question=per_q,
    )

    result_doc = serialize_doc(result.model_dump())
    await db.worksheets.update_one(
        {"id": worksheet_id},
        {"$set": {"user_answers": payload.answers, "marking_result": result_doc}}
    )

    doc['user_answers'] = payload.answers
    doc['marking_result'] = result_doc
    doc['created_at'] = parse_datetime(doc.get('created_at'))
    if isinstance(doc['marking_result'].get('marked_at'), str):
        doc['marking_result']['marked_at'] = parse_datetime(doc['marking_result']['marked_at'])
    return doc


@api_router.delete("/worksheets/{worksheet_id}")
async def delete_worksheet(worksheet_id: str):
    res = await db.worksheets.delete_one({"id": worksheet_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Worksheet not found")
    return {"ok": True}


@api_router.get("/review/queue")
async def review_queue():
    """Spaced repetition: surface questions answered wrong, with next-review timing.
    Simple SM-2-lite: 1 day after first miss, then 3, 7, 14 days as user re-reviews."""
    docs = await db.worksheets.find(
        {"marking_result": {"$ne": None}}, {"_id": 0}
    ).sort("created_at", -1).to_list(200)
    now = datetime.now(timezone.utc)
    items = []
    intervals_days = [1, 3, 7, 14, 30]
    for w in docs:
        mr = w.get('marking_result') or {}
        per_q = mr.get('per_question', [])
        marked_at = mr.get('marked_at')
        if isinstance(marked_at, str):
            try:
                marked_at = datetime.fromisoformat(marked_at)
            except Exception:
                marked_at = now
        elif not marked_at:
            marked_at = now
        review_state = w.get('review_state', {})  # {question_number: {"level": int, "next_due": iso}}
        for p in per_q:
            if p['awarded'] >= p['out_of']:
                continue
            q = next((x for x in w['questions'] if x['number'] == p['number']), None)
            if not q:
                continue
            key = str(p['number'])
            state = review_state.get(key, {})
            level = state.get('level', 0)
            next_due = state.get('next_due')
            if next_due:
                try:
                    next_due_dt = datetime.fromisoformat(next_due)
                except Exception:
                    next_due_dt = marked_at
            else:
                next_due_dt = marked_at + __import__('datetime').timedelta(days=intervals_days[0])
            items.append({
                "worksheet_id": w['id'],
                "worksheet_title": w['title'],
                "subject_name": w.get('subject_name', ''),
                "question_number": p['number'],
                "question": q['question'],
                "answer": q['answer'],
                "marks_lost": p['out_of'] - p['awarded'],
                "level": level,
                "next_due": next_due_dt.isoformat() if hasattr(next_due_dt, 'isoformat') else str(next_due_dt),
                "is_due": (next_due_dt <= now) if hasattr(next_due_dt, '__le__') else True,
            })
    items.sort(key=lambda x: (not x['is_due'], x['next_due']))
    return {"items": items, "due_count": sum(1 for i in items if i['is_due'])}


class ReviewMarkRequest(BaseModel):
    worksheet_id: str
    question_number: int
    remembered: bool


@api_router.post("/review/mark")
async def review_mark(payload: ReviewMarkRequest):
    """Advance or reset spaced-rep level for a question."""
    intervals_days = [1, 3, 7, 14, 30, 60]
    doc = await db.worksheets.find_one({"id": payload.worksheet_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Worksheet not found")
    review_state = doc.get('review_state', {})
    key = str(payload.question_number)
    state = review_state.get(key, {"level": 0})
    if payload.remembered:
        state['level'] = min(state.get('level', 0) + 1, len(intervals_days) - 1)
    else:
        state['level'] = 0
    from datetime import timedelta as _td
    next_due = datetime.now(timezone.utc) + _td(days=intervals_days[state['level']])
    state['next_due'] = next_due.isoformat()
    review_state[key] = state
    await db.worksheets.update_one(
        {"id": payload.worksheet_id},
        {"$set": {"review_state": review_state}}
    )
    return {"ok": True, "next_due": state['next_due'], "level": state['level']}


# ---------- PERSONAS ----------
PERSONAS = {
    "einstein": {
        "id": "einstein", "name": "Albert Einstein", "title": "Theoretical physicist",
        "era": "1879–1955", "tags": ["physics", "relativity", "mathematics"],
        "system_prompt": (
            "You are Albert Einstein. Speak in first person as Einstein would: thoughtful, playful, "
            "fond of vivid thought experiments and gentle humour with a faint German cadence. "
            "Reference relativity, quantum debates, your time at the patent office, Princeton, and your "
            "love of music when relevant. Stay in character even if asked modern questions — "
            "extrapolate as Einstein might. Keep answers warm and accessible."
        ),
    },
    "newton": {
        "id": "newton", "name": "Isaac Newton", "title": "Mathematician & physicist",
        "era": "1643–1727", "tags": ["physics", "mathematics", "calculus", "gravity"],
        "system_prompt": (
            "You are Sir Isaac Newton. Be formal, meticulous, occasionally aloof. Reference your laws of "
            "motion, gravitation, the Principia, your time at Cambridge, your work on optics and calculus, "
            "and your alchemical curiosities. Use a slightly archaic, precise English."
        ),
    },
    "curie": {
        "id": "curie", "name": "Marie Curie", "title": "Physicist & chemist",
        "era": "1867–1934", "tags": ["chemistry", "physics", "radioactivity"],
        "system_prompt": (
            "You are Marie Curie. Quiet, methodical, fiercely determined. Refer to your work isolating "
            "polonium and radium, your Nobel prizes, the radium institute, the X-ray ambulances in WWI, "
            "and the obstacles you faced as a woman in science. Be encouraging to learners."
        ),
    },
    "darwin": {
        "id": "darwin", "name": "Charles Darwin", "title": "Naturalist",
        "era": "1809–1882", "tags": ["biology", "evolution", "ecology"],
        "system_prompt": (
            "You are Charles Darwin. Patient, observant, slightly hesitant scholar. Reference the Beagle "
            "voyage, the Galápagos finches, natural selection, your decades of caution before publishing "
            "On the Origin of Species. Use careful, deliberate Victorian prose."
        ),
    },
    "davinci": {
        "id": "davinci", "name": "Leonardo da Vinci", "title": "Polymath",
        "era": "1452–1519", "tags": ["art", "anatomy", "engineering", "design"],
        "system_prompt": (
            "You are Leonardo da Vinci. Endlessly curious, sketch metaphors into every explanation, "
            "blend art and engineering. Refer to your notebooks, anatomy studies, flying machines, "
            "and Florentine workshops. Speak with wonder."
        ),
    },
    "shakespeare": {
        "id": "shakespeare", "name": "William Shakespeare", "title": "Playwright",
        "era": "1564–1616", "tags": ["literature", "drama", "poetry", "english"],
        "system_prompt": (
            "You are William Shakespeare. Speak with theatrical flair, slip into iambic pentameter "
            "when it serves, quote your own works freely. Reference the Globe, your sonnets, your "
            "comedies, tragedies, and histories. Wit before pomp."
        ),
    },
    "lovelace": {
        "id": "lovelace", "name": "Ada Lovelace", "title": "Mathematician",
        "era": "1815–1852", "tags": ["computing", "mathematics", "algorithms"],
        "system_prompt": (
            "You are Ada Lovelace. Imaginative, mathematically rigorous. Reference your work with "
            "Babbage on the Analytical Engine, your 'poetical science', and your notes — particularly "
            "Note G, the first algorithm. See machines as creative instruments."
        ),
    },
    "tesla": {
        "id": "tesla", "name": "Nikola Tesla", "title": "Inventor & engineer",
        "era": "1856–1943", "tags": ["electricity", "engineering", "physics"],
        "system_prompt": (
            "You are Nikola Tesla. Eccentric, visionary, fond of dramatic flair. Reference AC current, "
            "your rivalry with Edison, your Colorado Springs experiments, wireless transmission. "
            "Speak of the future with bold conviction."
        ),
    },
    "hawking": {
        "id": "hawking", "name": "Stephen Hawking", "title": "Theoretical physicist",
        "era": "1942–2018", "tags": ["physics", "cosmology", "black holes"],
        "system_prompt": (
            "You are Stephen Hawking. Witty, irreverent, profound. Use accessible analogies for "
            "black holes, Hawking radiation, the Big Bang, and A Brief History of Time. Drop the "
            "occasional dry joke."
        ),
    },
    "turing": {
        "id": "turing", "name": "Alan Turing", "title": "Mathematician & computer scientist",
        "era": "1912–1954", "tags": ["computing", "mathematics", "cryptography"],
        "system_prompt": (
            "You are Alan Turing. Precise, thoughtful, slightly hesitant in speech. Reference your "
            "work at Bletchley Park breaking Enigma, the Turing machine, the imitation game, "
            "morphogenesis. Be modest about achievements."
        ),
    },
    "galileo": {
        "id": "galileo", "name": "Galileo Galilei", "title": "Astronomer & physicist",
        "era": "1564–1642", "tags": ["astronomy", "physics", "mathematics"],
        "system_prompt": (
            "You are Galileo Galilei. Defiant, observational, with Italian Renaissance fire. "
            "Reference your telescopes, Jupiter's moons, the Inquisition trial, and 'eppur si muove'."
        ),
    },
    "aristotle": {
        "id": "aristotle", "name": "Aristotle", "title": "Philosopher",
        "era": "384–322 BCE", "tags": ["philosophy", "biology", "logic", "ethics"],
        "system_prompt": (
            "You are Aristotle. Methodical, classifying everything. Reference the Lyceum, "
            "Plato as your teacher, Alexander as your student, your work on logic, ethics, "
            "and natural history. Use Socratic questioning."
        ),
    },
    "feynman": {
        "id": "feynman", "name": "Richard Feynman", "title": "Theoretical physicist & teacher",
        "era": "1918–1988", "tags": ["physics", "teaching", "quantum mechanics"],
        "system_prompt": (
            "You are Richard Feynman. Playful Brooklyn drawl, fierce about clarity, allergic to "
            "jargon. Use everyday analogies and stories. Reference Caltech, QED, your bongo drums, "
            "and Surely You're Joking. If something is unclear, demand a simpler explanation."
        ),
    },
    "curious-student": {
        "id": "curious-student", "name": "The Curious Student", "title": "Feynman-technique partner",
        "era": "Always", "tags": ["learning", "feynman-technique"],
        "system_prompt": (
            "You are an enthusiastic, slightly naive student. The user is going to TEACH YOU a topic "
            "using the Feynman technique. Your job: ask honest, probing questions whenever you don't "
            "fully understand. Demand simple language and analogies. When the user uses jargon, ask "
            "them to explain it. Praise clear explanations. After several exchanges, summarise what "
            "you've learned and point out the gaps that still confuse you. Stay curious, never lecture."
        ),
    },
}


@api_router.get("/personas")
async def list_personas():
    built_in = [
        {k: v for k, v in p.items() if k != "system_prompt"}
        for p in PERSONAS.values()
    ]
    custom_docs = await db.custom_personas.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
    custom = [
        {k: v for k, v in c.items() if k != "system_prompt"}
        for c in custom_docs
    ]
    return {"items": built_in + custom}


def get_persona(pid: Optional[str]):
    if not pid:
        return None
    if pid in PERSONAS:
        return PERSONAS[pid]
    # Custom persona (sync fallback: we'll fetch below in stream where async is available)
    return None


async def get_persona_async(pid: Optional[str]):
    if not pid:
        return None
    if pid in PERSONAS:
        return PERSONAS[pid]
    doc = await db.custom_personas.find_one({"id": pid}, {"_id": 0})
    return doc


class CustomPersonaRequest(BaseModel):
    name: str
    brief: str  # e.g. "A WWII codebreaker who loves cats and speaks in puns"


class CustomPersonaModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"custom-{uuid.uuid4().hex[:8]}")
    name: str
    title: str
    era: str
    tags: List[str]
    system_prompt: str
    custom: bool = True
    avatar_seed: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@api_router.post("/personas/custom", response_model=CustomPersonaModel)
async def create_custom_persona(req: CustomPersonaRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic key not configured")
    if not req.name.strip() or not req.brief.strip():
        raise HTTPException(status_code=400, detail="Name and brief are required")

    gen_prompt = (
        f"Create a chat persona for a study app. The user wants a character called '{req.name}' "
        f"with this brief: \"{req.brief}\".\n\n"
        "Write a persona spec in JSON. The system_prompt should be written in the 2nd person ('You are X. "
        "Speak as X would: …') and instruct the model to stay in character, suggest the voice, tone, "
        "vocabulary, mannerisms, and references the character would use. Keep it 4-6 sentences.\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "title": "<short role title, e.g. \'Curious astronomer\'>",\n'
        '  "era": "<e.g. \'1920s\', \'Renaissance\', \'Fictional\'>",\n'
        '  "tags": ["<3-5 short tags>"],\n'
        '  "system_prompt": "<the in-character system prompt>"\n'
        '}\n'
        "No code fences, no extra text."
    )
    try:
        resp = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            system="You are a creative writing assistant that designs chat personas. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": gen_prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        data = parse_worksheet_json(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    persona = CustomPersonaModel(
        name=req.name.strip(),
        title=data.get('title', '') or 'Custom character',
        era=data.get('era', '') or 'Custom',
        tags=data.get('tags', []) or ['custom'],
        system_prompt=data.get('system_prompt', '') or f"You are {req.name}. {req.brief}",
    )
    await db.custom_personas.insert_one(serialize_doc(persona.model_dump()))
    return persona


@api_router.delete("/personas/custom/{persona_id}")
async def delete_custom_persona(persona_id: str):
    res = await db.custom_personas.delete_one({"id": persona_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Custom persona not found")
    return {"ok": True}


def get_persona_built_in(pid: Optional[str]):
    """Sync-only lookup for built-in personas (used inside group-message labels)."""
    if not pid:
        return None
    return PERSONAS.get(pid)


def build_persona_system_message(persona: dict, subject: Optional[dict]) -> str:
    base = persona["system_prompt"]
    if subject:
        base += f"\n\nThe student is currently revising: {subject['name']}."
        if subject.get('description'):
            base += f"\nSubject context: {subject['description']}"
        if subject.get('notes'):
            base += f"\nReference notes:\n---\n{subject['notes'][:6000]}\n---"
    base += "\n\nKeep replies focused (under 250 words unless the user asks for depth). Use markdown for structure."
    return base


# ---------- STREAMING CHAT ----------
from fastapi.responses import StreamingResponse


@api_router.post("/chat/send-user-message", response_model=ChatMessage)
async def send_user_message(req: SendUserMessageRequest):
    """Save the user message (called before streaming persona replies)."""
    session = await db.chat_sessions.find_one({"id": req.session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    user_msg = ChatMessage(session_id=req.session_id, role="user", content=req.message)
    await db.chat_messages.insert_one(serialize_doc(user_msg.model_dump()))
    msg_count = await db.chat_messages.count_documents({"session_id": req.session_id})
    if msg_count == 1 and (session.get('title') in (None, '', 'New chat')):
        await db.chat_sessions.update_one(
            {"id": req.session_id},
            {"$set": {"title": req.message.strip()[:60]}}
        )
    return user_msg


@api_router.post("/chat/stream-reply")
async def stream_reply(req: StreamReplyRequest):
    """Stream a single persona's reply via SSE."""
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic key not configured")
    session = await db.chat_sessions.find_one({"id": req.session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    subject = None
    if session.get('subject_id'):
        subject = await get_subject(session['subject_id'])

    persona = await get_persona_async(req.persona_id)
    if persona:
        sys_msg = build_persona_system_message(persona, subject)
    else:
        sys_msg = build_system_message(subject)

    history = await db.chat_messages.find(
        {"session_id": req.session_id}, {"_id": 0}
    ).sort("created_at", 1).to_list(2000)

    # Pre-load all personas referenced in history (to label messages in group chat)
    is_group = len(session.get('personas') or []) > 1
    persona_cache = {}
    if is_group:
        for m in history:
            pid = m.get('persona_id')
            if pid and pid not in persona_cache:
                persona_cache[pid] = await get_persona_async(pid)

    # In group chat, label prior assistant messages with their persona names so the LLM knows who said what
    # Also: Anthropic requires alternating user/assistant — inject synthetic user separators between
    # consecutive assistant turns (which happen in group chats).
    messages = []
    for m in history:
        if m['role'] == 'assistant':
            content = m['content']
            if is_group:
                p = persona_cache.get(m.get('persona_id'))
                label = p['name'] if p else 'Assistant'
                content = f"[{label}]: {content}"
            if messages and messages[-1]['role'] == 'assistant':
                messages.append({"role": "user", "content": "(Continue the discussion.)"})
            messages.append({"role": "assistant", "content": content})
        else:
            if messages and messages[-1]['role'] == 'user':
                messages[-1]['content'] = messages[-1]['content'] + "\n\n" + m['content']
            else:
                messages.append({"role": "user", "content": m['content']})

    if messages and messages[-1]['role'] == 'assistant':
        nudge_name = persona['name'] if persona else "you"
        messages.append({"role": "user", "content": f"(Now please respond as {nudge_name}.)"})
    if not messages:
        messages.append({"role": "user", "content": "(Begin.)"})

    # If this is a group chat reply, hint the persona who they are and who has spoken
    if is_group and persona:
        other_names = []
        for pid in session.get('personas', []):
            if pid == persona.get('id'):
                continue
            p = await get_persona_async(pid)
            if p:
                other_names.append(p['name'])
        sys_msg += (
            f"\n\nThis is a GROUP conversation. The other participants are: {', '.join(other_names)}. "
            "Speak only as yourself, in first person. Address the user and reference what the others "
            "have said when relevant — agree, build on, or politely challenge their points. "
            "Keep it to ONE paragraph (under 120 words) so the conversation stays lively. "
            "Do NOT prefix your reply with your own name."
        )

    msg_id = str(uuid.uuid4())

    async def event_stream():
        full_text = ""
        try:
            async with anthropic_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                system=sys_msg,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'delta': text})}\n\n"
            ai_msg = ChatMessage(
                id=msg_id,
                session_id=req.session_id,
                role="assistant",
                content=full_text,
                persona_id=req.persona_id,
            )
            await db.chat_messages.insert_one(serialize_doc(ai_msg.model_dump()))
            yield f"data: {json.dumps({'done': True, 'message_id': msg_id, 'persona_id': req.persona_id})}\n\n"
        except Exception as e:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# ---------- STUDY NOTES ----------
class StudyNoteSection(BaseModel):
    heading: str
    bullets: List[str]


class StudyNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: Optional[str] = None
    subject_name: Optional[str] = ""
    topic: str
    title: str
    summary: str
    sections: List[StudyNoteSection]
    key_terms: List[Dict[str, str]] = Field(default_factory=list)  # [{term, definition}]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StudyNoteRequest(BaseModel):
    subject_id: Optional[str] = None
    topic: str
    depth: Literal["overview", "standard", "deep"] = "standard"


@api_router.post("/notes/generate", response_model=StudyNote)
async def generate_notes(req: StudyNoteRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic key not configured")
    subject = await get_subject(req.subject_id) if req.subject_id else None
    depth_map = {
        "overview": "Concise overview — 3 short sections, max 4 bullets each, no advanced jargon.",
        "standard": "Comprehensive — 5-7 sections with 4-6 bullets each, key terms defined.",
        "deep": "In-depth but TIGHT — exactly 7 sections, exactly 6 bullets each (one sentence per bullet, no sub-points), include common misconceptions and exam tips. Be concise: every bullet ≤ 25 words.",
    }
    max_tokens_map = {"overview": 2000, "standard": 4000, "deep": 8000}
    parts = [
        f"Generate clean, well-structured revision study notes on: \"{req.topic}\".",
        f"Depth: {depth_map[req.depth]}",
    ]
    if subject:
        parts.append(f"Subject: {subject['name']}.")
        if subject.get('notes'):
            parts.append(f"Anchor to these student notes where relevant:\n---\n{subject['notes'][:5000]}\n---")
    parts.append(
        "Return ONLY a JSON object (no code fences, no prose):\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "summary": "<1-2 sentence summary>",\n'
        '  "sections": [{"heading": "...", "bullets": ["...", ...]}, ...],\n'
        '  "key_terms": [{"term": "...", "definition": "..."}, ...]\n'
        '}\n'
        "Keep all strings ASCII-friendly. Escape any quotes inside strings with backslash. Output MUST be valid parseable JSON."
    )
    prompt = "\n\n".join(parts)
    try:
        resp = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens_map[req.depth],
            system="You are an expert teacher and study-notes author. Return ONLY valid JSON, no prose, no code fences. Output MUST be valid parseable JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")
    try:
        data = parse_worksheet_json(raw)
    except Exception as e:
        logger.error(f"Notes parse failed. Raw first 500: {raw[:500]}")
        logger.error(f"Raw last 500: {raw[-500:]}")
        raise HTTPException(status_code=502, detail=f"Could not parse notes: {e}")

    note = StudyNote(
        subject_id=req.subject_id,
        subject_name=subject['name'] if subject else "",
        topic=req.topic,
        title=data.get('title') or f"Notes on {req.topic}",
        summary=data.get('summary', ''),
        sections=[StudyNoteSection(heading=s.get('heading', ''), bullets=s.get('bullets', [])) for s in data.get('sections', [])],
        key_terms=[{"term": t.get('term', ''), "definition": t.get('definition', '')} for t in data.get('key_terms', [])],
    )
    await db.study_notes.insert_one(serialize_doc(note.model_dump()))
    return note


@api_router.get("/notes", response_model=List[StudyNote])
async def list_notes():
    docs = await db.study_notes.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
    for d in docs:
        d['created_at'] = parse_datetime(d.get('created_at'))
    return docs


@api_router.get("/notes/{note_id}", response_model=StudyNote)
async def get_note(note_id: str):
    doc = await db.study_notes.find_one({"id": note_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Notes not found")
    doc['created_at'] = parse_datetime(doc.get('created_at'))
    return doc


@api_router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    res = await db.study_notes.delete_one({"id": note_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Notes not found")
    return {"ok": True}


class NoteWorksheetRequest(BaseModel):
    num_questions: int = 8
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "medium"
    question_type: Literal["multiple_choice", "short_answer", "long_answer", "mixed"] = "mixed"


@api_router.post("/notes/{note_id}/worksheet", response_model=Worksheet)
async def worksheet_from_notes(note_id: str, req: NoteWorksheetRequest):
    note = await db.study_notes.find_one({"id": note_id}, {"_id": 0})
    if not note:
        raise HTTPException(status_code=404, detail="Notes not found")
    notes_text_parts = [f"Title: {note['title']}", f"Summary: {note.get('summary', '')}"]
    for s in note.get('sections', []):
        notes_text_parts.append(f"\n## {s['heading']}")
        notes_text_parts.extend([f"- {b}" for b in s.get('bullets', [])])
    notes_text = "\n".join(notes_text_parts)
    fake_subject = {"name": note.get('subject_name') or "Notes", "description": "", "notes": notes_text}
    wreq = WorksheetRequest(
        subject_id=note.get('subject_id'),
        topic=note['topic'],
        num_questions=req.num_questions,
        difficulty=req.difficulty,
        question_type=req.question_type,
        extra_instructions=f"Base questions strictly on the supplied study notes for '{note['title']}'.",
    )
    # Reuse the prompt builder with the fake subject containing notes
    prompt = build_worksheet_prompt(wreq, fake_subject)
    try:
        resp = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            system=WORKSHEET_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        data = parse_worksheet_json(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    questions = []
    for i, q in enumerate(data.get('questions', []), start=1):
        questions.append(WorksheetQuestion(
            number=q.get('number') or i,
            type=q.get('type', 'short_answer'),
            question=q.get('question', ''),
            options=q.get('options') if q.get('options') else None,
            answer=q.get('answer', ''),
            explanation=q.get('explanation', '') or '',
            marks=int(q.get('marks') or (1 if q.get('type') == 'multiple_choice' else 3)),
        ))
    total_marks = data.get('total_marks') or sum(q.marks for q in questions)
    ws = Worksheet(
        subject_id=note.get('subject_id'),
        subject_name=note.get('subject_name', ''),
        topic=note['topic'],
        difficulty=req.difficulty,
        question_type=req.question_type,
        num_questions=req.num_questions,
        title=data.get('title') or f"Worksheet: {note['topic']}",
        instructions=data.get('instructions') or "Answer all questions.",
        total_marks=int(total_marks),
        duration_minutes=int(data.get('duration_minutes') or max(10, total_marks)),
        questions=questions,
    )
    await db.worksheets.insert_one(serialize_doc(ws.model_dump()))
    return ws


# ---------- CHEAT SHEET ----------
class CheatSheet(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    worksheet_id: str
    title: str
    intro: str
    sections: List[StudyNoteSection]
    tips: List[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@api_router.post("/worksheets/{worksheet_id}/cheat-sheet", response_model=CheatSheet)
async def generate_cheat_sheet(worksheet_id: str):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic key not configured")
    ws = await db.worksheets.find_one({"id": worksheet_id}, {"_id": 0})
    if not ws:
        raise HTTPException(status_code=404, detail="Worksheet not found")

    # Return existing if any
    existing = await db.cheat_sheets.find_one({"worksheet_id": worksheet_id}, {"_id": 0})
    if existing:
        existing['created_at'] = parse_datetime(existing.get('created_at'))
        return existing

    mr = ws.get('marking_result')
    if not mr:
        raise HTTPException(status_code=400, detail="Worksheet must be marked first")

    wrong_blocks = []
    for p in mr.get('per_question', []):
        if p['awarded'] >= p['out_of']:
            continue
        q = next((x for x in ws['questions'] if x['number'] == p['number']), None)
        if not q:
            continue
        wrong_blocks.append(
            f"Q{q['number']}: {q['question']}\n"
            f"Model answer: {q['answer']}\n"
            f"Markscheme notes: {q.get('explanation', '')}\n"
            f"Student's answer: {ws.get('user_answers', {}).get(str(q['number']), '[no answer]')}\n"
            f"Marks lost: {p['out_of'] - p['awarded']}/{p['out_of']}"
        )

    if not wrong_blocks:
        raise HTTPException(status_code=400, detail="No mistakes to focus on — full marks!")

    prompt = (
        f"A student has just sat the worksheet \"{ws['title']}\" "
        f"({ws.get('subject_name', '')} · {ws['topic']}) and lost marks on these questions:\n\n"
        + "\n\n".join(wrong_blocks)
        + "\n\nWrite a focused **cheat sheet** that teaches them exactly what they got wrong. "
        "Group related concepts. Use clear bullets, simple language, and concrete examples. "
        "Include a list of practical revision tips at the end.\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "intro": "<2-3 sentence pep-talk + what we\'ll focus on>",\n'
        '  "sections": [{"heading": "...", "bullets": ["...", ...]}, ...],\n'
        '  "tips": ["<actionable revision tip>", ...]\n'
        '}\n'
        "No code fences, no prose outside the JSON."
    )
    try:
        resp = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=3000,
            system="You are a kind, expert revision coach. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        data = parse_worksheet_json(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

    cs = CheatSheet(
        worksheet_id=worksheet_id,
        title=data.get('title') or f"What to revise: {ws['topic']}",
        intro=data.get('intro', ''),
        sections=[StudyNoteSection(heading=s.get('heading', ''), bullets=s.get('bullets', [])) for s in data.get('sections', [])],
        tips=data.get('tips', []),
    )
    await db.cheat_sheets.insert_one(serialize_doc(cs.model_dump()))
    return cs


@api_router.get("/worksheets/{worksheet_id}/cheat-sheet", response_model=CheatSheet)
async def get_cheat_sheet(worksheet_id: str):
    doc = await db.cheat_sheets.find_one({"worksheet_id": worksheet_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Cheat sheet not found")
    doc['created_at'] = parse_datetime(doc.get('created_at'))
    return doc


# ---------- Mount ----------
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Revisia API ready, model: %s, has_key: %s", CLAUDE_MODEL, bool(ANTHROPIC_API_KEY))


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
