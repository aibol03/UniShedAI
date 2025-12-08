import os
import time
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

app = FastAPI()

# Разрешаем запросы с любого сайта (важно для фронтенда)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- НАСТРОЙКА API KEY ---
# Код ищет ключ в переменных системы (для Render). 
# Если запускаешь локально — вставь ключ во вторые кавычки.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- МОДЕЛИ ДАННЫХ ---
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
    days: List[str]
    times: List[str]

class ChatRequest(BaseModel):
    question: str
    schedule_context: str
    lang: str

# --- АЛГОРИТМ ГЕНЕРАЦИИ ---
@app.post("/generate")
def generate_schedule(data: SessionData):
    slots = []
    slot_map = {}
    
    # 1. Создаем временные слоты
    for d in data.days:
        for t_idx, t in enumerate(data.times):
            s_name = f"{d} {t}"
            slots.append(s_name)
            slot_map[s_name] = {"day": d, "time_idx": t_idx}

    if not slots:
        return {"status": "error", "message": "Не заданы дни или время звонков!", "schedule": [], "errors": []}

    # Справочники для быстрого доступа
    group_sizes = {g.name: g.size for g in data.groups}
    room_caps = {r.name: r.capacity for r in data.rooms}
    room_names = [r.name for r in data.rooms]
    
    # Находим самый большой кабинет (для разбиения потоков)
    max_cap = 0
    if data.rooms:
        max_cap = max(r.capacity for r in data.rooms)
        if max_cap == 0: max_cap = 9999

    tasks = []
    
    # 2. Обработка нагрузки (Создание задач)
    for assign in data.assignments:
        count = assign.count
        
        if assign.type == 'seminar':
            # СЕМИНАРЫ: Разбиваем каждую группу отдельно
            for group_name in assign.groups:
                for _ in range(count):
                    tasks.append({
                        "teacher": assign.teacher,
                        "groups": [group_name],
                        "subject": assign.subject,
                        "type": "seminar"
                    })
        else:
            # ЛЕКЦИИ: Умные потоки (Smart Streams)
            # Сортируем группы по размеру
            sorted_groups = sorted(assign.groups, key=lambda g: group_sizes.get(g, 0), reverse=True)
            current_stream = []
            current_size = 0
            streams = []
            
            for g_name in sorted_groups:
                g_size = group_sizes.get(g_name, 0)
                # Если группа огромная - отдельный поток
                if g_size > max_cap:
                    streams.append([g_name])
                    continue

                # Пытаемся добавить в текущий поток
                if current_size + g_size <= max_cap:
                    current_stream.append(g_name)
                    current_size += g_size
                else:
                    # Поток переполнен, сохраняем и начинаем новый
                    if current_stream: streams.append(current_stream)
                    current_stream = [g_name]
                    current_size = g_size
            
            if current_stream: streams.append(current_stream)
            
            # Создаем задачи для потоков
            for s in streams:
                for _ in range(count):
                    tasks.append({
                        "teacher": assign.teacher,
                        "groups": s,
                        "subject": assign.subject,
                        "type": "lecture"
                    })

    # Сортировка задач: Сначала Лекции, потом самые большие группы
    tasks.sort(key=lambda x: (x['type'] == 'lecture', len(x['groups']), sum(group_sizes.get(g,0) for g in x['groups'])), reverse=True)

    schedule = []
    errors = []
    
    # Словари занятости
    occ_teachers = {s: set() for s in slots}
    occ_groups = {s: set() for s in slots}
    occ_rooms = {s: set() for s in slots}
    teacher_daily = {t: {d: set() for d in data.days} for t in data.teachers}

    # 3. Расстановка (Жадный алгоритм)
    for task in tasks:
        t_name = task["teacher"]
        t_groups = task["groups"]
        stud_count = sum(group_sizes.get(g, 0) for g in t_groups)

        # Эвристика: Анти-окно (ищем слоты рядом с уже занятыми)
        def score(slot):
            info = slot_map[slot]
            exist = teacher_daily[t_name][info["day"]]
            if not exist: return 10 # Новый день - хорошо
            dist = min(abs(info["time_idx"] - e) for e in exist)
            if dist == 1: return 100 # Подряд - отлично
            if dist == 2: return -50 # Окно - плохо
            return 0

        # Фильтр доступных слотов
        avail = []
        for slot in slots:
            if t_name in occ_teachers[slot]: continue # Учитель занят
            if any(g in occ_groups[slot] for g in t_groups): continue # Группа занята
            if t_name in data.teacher_busy and slot in data.teacher_busy[t_name]: continue # График учителя
            avail.append(slot)
        
        # Сортируем слоты по качеству
        avail.sort(key=score, reverse=True)

        placed = False
        reason = "Нет общего свободного времени"
        
        for slot in avail:
            target = None
            def fits(r): return (room_caps.get(r, 9999) or 9999) >= stud_count

            # Привязка (Только если НЕ лекция)
            if task['type'] != 'lecture' and t_name in data.teacher_prefs:
                pref = data.teacher_prefs[t_name]
                if pref not in occ_rooms[slot] and fits(pref): target = pref
            
            # Поиск лучшего кабинета
            if not target:
                free = [r for r in room_names if r not in occ_rooms[slot]]
                # Сортируем: сначала маленькие подходящие
                free.sort(key=lambda r: room_caps.get(r, 9999))
                for r in free:
                    if fits(r): 
                        target = r; break
            
            if target:
                info = slot_map[slot]
                time_part = slot.replace(info["day"], "").strip()
                
                schedule.append({
                    "slot": slot, "day": info["day"], "time": time_part,
                    "teacher": t_name, "group": ", ".join(t_groups),
                    "subject": task["subject"], "room": target, "type": task["type"]
                })
                
                # Блокируем ресурсы
                occ_teachers[slot].add(t_name)
                occ_rooms[slot].add(target)
                for g in t_groups: occ_groups[slot].add(g)
                teacher_daily[t_name][info["day"]].add(info["time_idx"])
                placed = True
                break
            else:
                reason = "Нет свободных кабинетов нужной вместимости"
        
        if not placed:
            errors.append(f"{task['subject']} ({task['type']}) для {t_name}. Причина: {reason}")

    # Сортировка результата по времени
    schedule.sort(key=lambda x: slots.index(x["slot"]))
    return {"status": "success", "schedule": schedule, "errors": errors}

# --- ИИ ПОМОЩНИК ---
@app.post("/ask_ai")
def ask_ai(payload: ChatRequest):
    lang_sys = "Жауапты қазақ тілінде бер." if payload.lang == 'kz' else "Отвечай на русском."
    
    prompt = f"""
    {lang_sys}
    Ты — интеллектуальный диспетчер университета.
    Данные расписания (JSON): {payload.schedule_context}
    Вопрос пользователя: "{payload.question}"
    
    ИНСТРУКЦИЯ:
    1. ОПРЕДЕЛИ ЯЗЫК вопроса. Отвечай на нем.
    
    2. ЕСЛИ ПРОСЯТ ФАЙЛ (Excel, CSV, скачать, файл):
       - Сгенерируй CSV. Заголовки на языке вопроса.
       - Оберни данные в теги: 
       :::csv_start:::
       (данные)
       :::csv_end:::
       
    3. ЕСЛИ ПРОСЯТ ТАБЛИЦУ (или изменить вид):
       - Создай Markdown таблицу.
       - Если просят "дни в столбцах" — переверни таблицу.
       
    4. Иначе — краткий текстовый ответ.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash') 
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"AI Error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
