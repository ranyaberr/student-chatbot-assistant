# --------------------------------------------------------- 
# MEMORY: Helper functions for student memory (memories.json) 
# --------------------------------------------------------- 
import os 
import json 
import uuid 
 
 
from conversations import safe_load_json 
 
DATA_FOLDER = "data" 
MEMORY_FILE = os.path.join(DATA_FOLDER, "memories.json") 
 
def load_memories(): 
    return safe_load_json(MEMORY_FILE, []) 
 
def save_memories(memories): 
    os.makedirs(DATA_FOLDER, exist_ok=True) 
    with open(MEMORY_FILE, "w", encoding="utf-8") as f: 
        json.dump(memories, f, indent=2, ensure_ascii=False) 
 
def extract_memory_from_exchange(user_message, assistant_message , client , model_name): 
    """ 
    Ask the LLM whether this exchange contains durable info worth 
    remembering. Returns a list of fact strings (possibly empty). 
    """ 
    extraction_prompt = f"""Analyze this exchange between a student and an assistant. 
 
Extract ONLY durable, reusable facts about the STUDENT that would help 
in future, unrelated conversations. Examples of what to extract: 
- current university modules 
- current projects 
- technologies/programming languages being learned 
- preferred explanation style 
- learning goals 
- study habits 
- recurring interests 
- future objectives 
- important preferences 
 
Do NOT extract: 
- one-time questions with no future relevance (e.g. "What is a Python loop?") 
- greetings, jokes, small talk 
- temporary/throwaway requests 
 
If nothing is worth remembering, return an empty JSON array: [] 
 
Respond ONLY with a JSON array of short fact strings. Example: 
["Studying 2nd-year engineering", "Currently learning React for a school project"] 
 
User message: {user_message} 
Assistant message: {assistant_message} 
""" 
 
    response = client.chat.completions.create( 
        model=model_name, 
        messages=[{"role": "user", "content": extraction_prompt}] 
    ) 
 
    raw_reply = response.choices[0].message.content.strip() 
 
    if raw_reply.startswith("```"): 
        raw_reply = raw_reply.strip("`") 
        if raw_reply.startswith("json"): 
            raw_reply = raw_reply[4:] 
        raw_reply = raw_reply.strip() 
 
    try: 
        facts = json.loads(raw_reply) 
        if not isinstance(facts, list): 
            return [] 
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()] 
    except json.JSONDecodeError: 
        return [] 
 
def classify_facts_batch(new_facts, existing_memories, client, model_name): 
    """ 
    Ask the LLM, in a single call, whether each fact in new_facts 
    duplicates/updates an existing memory. Returns a list of 
    (action, index) tuples, one per fact in new_facts, in the same 
    order. action is "skip", "update", or "add" - identical semantics 
    to the previous per-fact is_duplicate_or_similar(). 
    """ 
    if not new_facts: 
        return [] 
 
    if not existing_memories: 
        return [("add", None) for _ in new_facts] 
 
    memory_list_text = "\n".join( 
        [f"{i}: {m['text']}" for i, m in enumerate(existing_memories)] 
    ) 
 
    facts_list_text = "\n".join( 
        [f"{i}: {fact}" for i, fact in enumerate(new_facts)] 
    ) 
 
    check_prompt = f"""Here is a list of existing memories about a student: 
{memory_list_text} 
 
Here is a list of NEW facts to consider, each identified by an index: 
{facts_list_text} 
 
For EACH new fact, decide one of: 
- "skip" if this fact is already covered by an existing memory (true duplicate) 
- "update:<index>" if this fact updates/replaces an existing memory (e.g. new module replaces old one) 
- "add" if this is genuinely new information 
 
Respond ONLY with a JSON array with exactly one decision per new fact, 
in the same order as the new facts list above. Each element must be a 
string: "skip", "update:<index>", or "add". Example for 3 new facts: 
["add", "skip", "update:2"] 
""" 
 
    response = client.chat.completions.create( 
        model=model_name, 
        messages=[{"role": "user", "content": check_prompt}] 
    ) 
 
    raw_reply = response.choices[0].message.content.strip() 
 
    if raw_reply.startswith("```"): 
        raw_reply = raw_reply.strip("`") 
        if raw_reply.startswith("json"): 
            raw_reply = raw_reply[4:] 
        raw_reply = raw_reply.strip() 
 
    try: 
        decisions_raw = json.loads(raw_reply) 
        if not isinstance(decisions_raw, list): 
            return [("add", None) for _ in new_facts] 
    except json.JSONDecodeError: 
        return [("add", None) for _ in new_facts] 
 
    results = [] 
    for i in range(len(new_facts)): 
        if i >= len(decisions_raw) or not isinstance(decisions_raw[i], str): 
            results.append(("add", None)) 
            continue 
 
        decision = decisions_raw[i].strip().lower() 
 
        if decision.startswith("skip"): 
            results.append(("skip", None)) 
        elif decision.startswith("update"): 
            try: 
                index = int(decision.split(":")[1].strip()) 
                if 0 <= index < len(existing_memories): 
                    results.append(("update", index)) 
                else: 
                    results.append(("add", None)) 
            except (IndexError, ValueError): 
                results.append(("add", None)) 
        else: 
            results.append(("add", None)) 
 
    return results 
 
def store_new_memories(new_facts, conversation_id, client , model_name): 
    """Add new facts to memories.json, handling duplicates/updates.""" 
    if not new_facts: 
        return 
 
    memories = load_memories() 
 
    decisions = classify_facts_batch(new_facts, memories, client, model_name) 
 
    for fact, (action, index) in zip(new_facts, decisions): 
        if action == "skip": 
            continue 
        elif action == "update": 
            memories[index]["text"] = fact 
            memories[index]["conversation_id"] = conversation_id 
        else:  # add 
            memories.append({ 
                "id": str(uuid.uuid4()), 
                "text": fact, 
                "conversation_id": conversation_id 
            }) 
 
    save_memories(memories) 
 
def retrieve_relevant_memories(current_question, all_memories,client,model_name , max_memories=5): 
    """ 
    Ask the LLM which stored memories are relevant to the current question. 
    Returns a list of relevant memory text strings. 
    """ 
    if not all_memories: 
        return [] 
 
    memory_list_text = "\n".join( 
        [f"{i}: {m['text']}" for i, m in enumerate(all_memories)] 
    ) 
 
    retrieval_prompt = f"""Here is a list of stored memories about a student: 
{memory_list_text} 
 
Current question from the student: "{current_question}" 
 
Which memories (if any) are relevant to understanding or answering this 
question? Relevant means it would help personalize or inform the response. 
 
Respond ONLY with a JSON array of the relevant indices (numbers). 
Example: [0, 2] 
If none are relevant, respond with: [] 
""" 
 
    response = client.chat.completions.create( 
        model=model_name, 
        messages=[{"role": "user", "content": retrieval_prompt}] 
    ) 
 
    raw_reply = response.choices[0].message.content.strip() 
 
    if raw_reply.startswith("```"): 
        raw_reply = raw_reply.strip("`") 
        if raw_reply.startswith("json"): 
            raw_reply = raw_reply[4:] 
        raw_reply = raw_reply.strip() 
 
    try: 
        indices = json.loads(raw_reply) 
        if not isinstance(indices, list): 
            return [] 
        relevant = [] 
        for i in indices[:max_memories]: 
            if isinstance(i, int) and 0 <= i < len(all_memories): 
                relevant.append(all_memories[i]["text"]) 
        return relevant 
    except json.JSONDecodeError: 
        return [] 