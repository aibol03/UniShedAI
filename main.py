import os
import random
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "ВСТАВЬ_СЮДА_СВОЙ_GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class GroupItem(BaseModel):
    name: str
    size: int = 0

class RoomItem(BaseModel):
    name: str
    capacity: int = 9999

class Assignment(BaseModel):
    teacher: str
    groups: List[str]
    subject: str
    count: int
    type: str 

class SessionData(BaseModel):
    teachers: List[str]
    subjects: List[str] = []
    groups: List[GroupItem]
    rooms: List[RoomItem]
    assignments: List[Assignment]
    teacher_prefs: Dict[str, str] = {}
    teacher_busy: Dict[str, List[str]] = {}
    room_busy: Dict[str, List[str]] = {} 
    group_busy: Dict[str, List[str]] = {}
    days: List[str]
    times: List[str]

class ChatRequest(BaseModel):
    question: str
    schedule_context: str
    lang: str

@app.post("/generate")
def generate_schedule(data: SessionData):
    slots = []
    slot_map = {}
    
    for d in data.days:
        for t_idx, t in enumerate(data.times):
            s_name = f"{d} {t}"
            slots.append(s_name)
            slot_map[s_name] = {"day": d, "time_idx": t_idx}

    if not slots:
        return {"status": "error", "message": "Не заданы дни или время звонков!", "schedule": [], "errors": []}

    group_sizes = {g.name: g.size for g in data.groups}
    room_caps = {r.name: r.capacity for r in data.rooms}
    room_names = [r.name for r in data.rooms]
    
    max_cap = 0
    if data.rooms:
        max_cap = max(r.capacity for r in data.rooms)
        if max_cap == 0: max_cap = 9999

    tasks = []
    
    for assign in data.assignments:
        count = assign.count
        if assign.type == 'seminar':
            for group_name in assign.groups:
                for _ in range(count):
                    tasks.append({
                        "teacher": assign.teacher, "groups": [group_name],
                        "subject": assign.subject, "type": "seminar"
                    })
        else:
            sorted_groups = sorted(assign.groups, key=lambda g: group_sizes.get(g, 0), reverse=True)
            current_stream = []
            current_size = 0
            streams = []
            
            for g_name in sorted_groups:
                g_size = group_sizes.get(g_name, 0)
                if g_size > max_cap:
                    streams.append([g_name])
                    continue
                if current_size + g_size <= max_cap:
                    current_stream.append(g_name)
                    current_size += g_size
                else:
                    if current_stream: streams.append(current_stream)
                    current_stream = [g_name]
                    current_size = g_size
            
            if current_stream: streams.append(current_stream)
            
            for s in streams:
                for _ in range(count):
                    tasks.append({
                        "teacher": assign.teacher, "groups": s,
                        "subject": assign.subject, "type": "lecture"
                    })

    tasks.sort(key=lambda x: (x['type'] == 'lecture', len(x['groups']), sum(group_sizes.get(g,0) for g in x['groups'])), reverse=True)

    best_schedule = []
    best_errors = ["dummy"] * 9999
    attempts = 50 if len(tasks) < 100 else 20 
    
    for attempt in range(attempts):
        schedule = []
        errors = []
        
        occ_teachers = {s: set() for s in slots}
        occ_groups = {s: set() for s in slots}
        occ_rooms = {s: set() for s in slots}
        teacher_daily = {t: {d: set() for d in data.days} for t in data.teachers}
        
        current_tasks = tasks.copy()
        if attempt > 0:
            random.shuffle(current_tasks)
            current_tasks.sort(key=lambda x: x['type'] == 'lecture', reverse=True)

        for task in current_tasks:
            t_name = task["teacher"]
            t_groups = task["groups"]
            stud_count = sum(group_sizes.get(g, 0) for g in t_groups)

            def score(slot):
                info = slot_map[slot]
                exist = teacher_daily[t_name][info["day"]]
                if not exist: return 10 
                dist = min(abs(info["time_idx"] - e) for e in exist)
                if dist == 1: return 100 
                if dist == 2: return -50 
                return 0

            avail = []
            for slot in slots:
                if t_name in occ_teachers[slot]: continue
                if any(g in occ_groups[slot] for g in t_groups): continue
                if t_name in data.teacher_busy and slot in data.teacher_busy[t_name]: continue
                if any((g in data.group_busy and slot in data.group_busy[g]) for g in t_groups): continue
                avail.append(slot)
            
            avail.sort(key=lambda s: score(s) + (random.random() if attempt > 0 else 0), reverse=True)

            placed = False
            reason = "Бос уақыт сәйкес келмеді (Оқытушы немесе Топ бос емес)"
            
            for slot in avail:
                target = None
                def fits(r): return (room_caps.get(r, 9999) or 9999) >= stud_count

                if task['type'] != 'lecture' and t_name in data.teacher_prefs:
                    pref = data.teacher_prefs[t_name]
                    is_room_busy = (pref in data.room_busy) and (slot in data.room_busy[pref])
                    if pref not in occ_rooms[slot] and fits(pref) and not is_room_busy: 
                        target = pref
                
                if not target:
                    free = [r for r in room_names if r not in occ_rooms[slot]]
                    free.sort(key=lambda r: room_caps.get(r, 9999))
                    for r in free:
                        is_room_busy = (r in data.room_busy) and (slot in data.room_busy[r])
                        if fits(r) and not is_room_busy: 
                            target = r; break
                
                if target:
                    info = slot_map[slot]
                    time_part = slot.replace(info["day"], "").strip()
                    
                    schedule.append({
                        "slot": slot, "day": info["day"], "time": time_part,
                        "teacher": t_name, "group": ", ".join(t_groups),
                        "subject": task["subject"], "room": target, "type": task["type"]
                    })
                    
                    occ_teachers[slot].add(t_name)
                    occ_rooms[slot].add(target)
                    for g in t_groups: occ_groups[slot].add(g)
                    teacher_daily[t_name][info["day"]].add(info["time_idx"])
                    placed = True
                    break
                else:
                    reason = "Сыйымдылығы жететін бос кабинет жоқ"
            
            if not placed:
                errors.append(f"{task['subject']} ({task['type']}) -> {t_name}, Топ: {', '.join(t_groups)}. Себеп: {reason}")
        
        if len(errors) < len(best_errors):
            best_schedule = schedule
            best_errors = errors
            
        if len(best_errors) == 0:
            break

    best_schedule.sort(key=lambda x: slots.index(x["slot"]))
    return {"status": "success", "schedule": best_schedule, "errors": best_errors}

@app.post("/ask_ai")
def ask_ai(payload: ChatRequest):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        You are a helpful AI schedule assistant. Answer the user question based on the schedule context provided.
        User Language: {payload.lang}
        Current Schedule Data:
        {payload.schedule_context}
        
        User Question: {payload.question}
        """
        response = model.generate_content(prompt)
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
