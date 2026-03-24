"""
CareVoice AI — Vector RAG Index Setup

Creates Azure AI Search index with vector fields,
generates embeddings via Azure OpenAI text-embedding-ada-002,
and uploads elder care protocol documents with vectors.

Run once:  python setup_search_index.py
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "carevoice-protocols")

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
OPENAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002-2")

API_VERSION_SEARCH = "2024-07-01"
API_VERSION_OPENAI = "2024-10-21"

HEADERS_SEARCH = {"api-key": SEARCH_KEY, "Content-Type": "application/json"}
HEADERS_OPENAI = {"api-key": OPENAI_KEY, "Content-Type": "application/json"}

VECTOR_DIM = 1536  # text-embedding-ada-002 output dimension


# ─── Step 1: Create Index with Vector Field ──────────────────────────────────

INDEX_SCHEMA = {
    "name": INDEX_NAME,
    "fields": [
        {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
        {"name": "title", "type": "Edm.String", "searchable": True, "filterable": True},
        {"name": "category", "type": "Edm.String", "searchable": True, "filterable": True, "facetable": True},
        {"name": "content", "type": "Edm.String", "searchable": True},
        {"name": "source", "type": "Edm.String", "filterable": True},
        {"name": "keywords", "type": "Collection(Edm.String)", "searchable": True, "filterable": True},
        {
            "name": "contentVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "dimensions": VECTOR_DIM,
            "vectorSearchProfile": "default-profile",
        },
    ],
    "vectorSearch": {
        "algorithms": [
            {
                "name": "default-algorithm",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            }
        ],
        "profiles": [
            {
                "name": "default-profile",
                "algorithm": "default-algorithm",
            }
        ],
    },
    "suggesters": [
        {"name": "sg", "searchMode": "analyzingInfixMatching", "sourceFields": ["title", "category"]}
    ],
}


def create_index():
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION_SEARCH}"
    r = requests.put(url, headers=HEADERS_SEARCH, json=INDEX_SCHEMA)
    if r.status_code in (200, 201):
        print(f"[OK] Index '{INDEX_NAME}' created with vector search")
    else:
        print(f"[ERROR] Index creation: {r.status_code} — {r.text[:200]}")
    return r.status_code in (200, 201)


# ─── Step 2: Generate Embeddings ─────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    url = f"{OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={API_VERSION_OPENAI}"
    r = requests.post(url, headers=HEADERS_OPENAI, json={"input": text})
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


# ─── Step 3: Protocol Documents ──────────────────────────────────────────────

PROTOCOLS = [
    {
        "id": "proto-001",
        "title": "Daily Wellness Check-In Protocol",
        "category": "wellness_check",
        "source": "CareVoice Standard Operating Procedures v2.0",
        "keywords": ["wellness", "check-in", "daily", "protocol", "greeting", "assessment"],
        "content": """DAILY WELLNESS CHECK-IN PROTOCOL

PURPOSE: Conduct a structured yet conversational daily wellness assessment for elderly patients living independently.

PROCEDURE:
1. IDENTITY VERIFICATION: Before disclosing any patient information, verify the identity of the person answering the phone. Ask them to state their first name. Do not reveal the patient's name, medical information, or any protected health information until identity is confirmed. If identity cannot be verified after 2 attempts, end the call politely without disclosing any information.

2. GREETING PHASE: Use the patient's preferred name. Establish a warm, friendly tone. Reference previous interactions when available. Do not rush — allow the patient time to settle into the conversation.

3. ASSESSMENT AREAS (cover naturally, not as a checklist):
   a. Sleep Quality: Ask about last night's sleep. Note any changes in sleep patterns.
   b. Nutrition: Ask about recent meals. Note appetite changes or skipped meals.
   c. Medication Adherence: Ask if medications were taken. If patient is unsure, provide their medication list from records. Never scold for missed doses.
   d. Physical Comfort: Ask about pain, mobility, falls, dizziness. Note any new symptoms.
   e. Emotional Wellbeing: Assess mood, loneliness, anxiety. Listen for signs of depression.
   f. Social Connection: Ask about recent contact with family, friends, caregivers.
   g. Cognitive Status: Note any confusion, memory issues, or disorientation during conversation.

4. CLOSING: Do not initiate goodbye. Wait for the patient to indicate they are ready to end the call. Summarize any concerns noted. Confirm next check-in time.

ESCALATION CRITERIA:
- Score <= 3 on any wellness dimension: Alert primary caregiver via SMS
- Fall reported: Immediate alert to caregiver and flag for physician review
- Chest pain or breathing difficulty: Advise calling 911 and alert emergency contacts
- Signs of cognitive decline: Flag for physician review within 24 hours
- Signs of abuse or neglect: Report per mandatory reporting requirements"""
    },
    {
        "id": "proto-002",
        "title": "Medication Management Protocol for Elderly Patients",
        "category": "medication",
        "source": "CareVoice Clinical Guidelines — Medication Management",
        "keywords": ["medication", "adherence", "elderly", "reminders", "side effects", "polypharmacy"],
        "content": """MEDICATION MANAGEMENT PROTOCOL

PURPOSE: Support medication adherence in elderly patients through conversational check-ins without providing medical advice.

SCOPE OF PRACTICE — CRITICAL:
- AI companions may REMIND patients of their prescribed medications as documented in their medical records
- AI companions may ASK whether medications were taken
- AI companions must NEVER recommend starting, stopping, or changing any medication
- AI companions must NEVER provide dosage advice beyond what is in the patient record
- AI companions must NEVER interpret symptoms as medication side effects
- If a patient reports concerning symptoms, advise them to contact their healthcare provider

PROCEDURE:
1. Ask naturally during conversation: "By the way, did you get a chance to take your medications today?"
2. If patient confirms: Acknowledge positively and move on.
3. If patient forgot: Respond supportively — "No worries at all! Would you like me to remind you what's on your list?"
4. If patient asks what medications they take: Read from their medical record EXACTLY as documented. Do not paraphrase dosages or frequencies.
5. If patient reports missing multiple days: Flag as medium concern for caregiver follow-up.
6. If patient reports side effects or adverse reactions: Do NOT interpret. Say: "I'm glad you told me. That's something your doctor would want to know about. Would you like me to let your care team know?"

COMMON MEDICATIONS IN ELDER CARE:
- Metformin (Type 2 diabetes): Usually taken with meals. Common side effects include stomach upset.
- Lisinopril (High blood pressure): Usually taken once daily in morning. May cause dizziness.
- Acetaminophen (Pain relief): Used as needed. Do not exceed recommended daily dose.
- Amlodipine (Blood pressure): Once daily. May cause ankle swelling.
- Omeprazole (Acid reflux): Usually taken before breakfast.

DOCUMENTATION: Log medication adherence status for each check-in. Track patterns of non-adherence for caregiver reporting."""
    },
    {
        "id": "proto-003",
        "title": "Fall Prevention and Response Protocol",
        "category": "safety",
        "source": "CareVoice Safety Protocols — Fall Management",
        "keywords": ["fall", "prevention", "response", "mobility", "safety", "emergency"],
        "content": """FALL PREVENTION AND RESPONSE PROTOCOL

PURPOSE: Detect fall risks and respond appropriately when a fall is reported during wellness check-ins.

RISK ASSESSMENT DURING CALLS:
- Ask about mobility: "Have you been moving around okay today?"
- Ask about dizziness: "Have you felt lightheaded or dizzy at all?"
- Ask about environment: "Is your home feeling safe and comfortable?"
- Note any mentions of tripping, stumbling, or unsteadiness

IF A FALL IS REPORTED:
1. Stay calm. Ask: "Are you hurt? Can you tell me where you are right now?"
2. If patient reports injury or inability to get up: Classify as CRITICAL. Advise: "I want to make sure you're safe. Can you call 911, or would you like me to alert your emergency contacts?"
3. If patient got up safely and is not injured: Classify as HIGH concern. Log the fall. Ask about circumstances.
4. Alert primary caregiver regardless of severity.
5. Flag for physician review — falls in elderly patients may indicate medication side effects, blood pressure issues, or neurological changes.

DOCUMENTATION: Record all fall reports with severity, circumstances, and response taken. Track fall frequency over time."""
    },
    {
        "id": "proto-004",
        "title": "Pain Assessment Protocol for Elderly Patients",
        "category": "pain_management",
        "source": "CareVoice Clinical Guidelines — Pain Assessment",
        "keywords": ["pain", "assessment", "elderly", "chronic", "acute", "comfort"],
        "content": """PAIN ASSESSMENT PROTOCOL

PURPOSE: Assess pain levels in elderly patients during wellness check-ins and escalate appropriately.

ASSESSMENT APPROACH:
- Ask open-ended questions: "How is your body feeling today? Any aches or pains?"
- If pain is reported, follow up: "Can you tell me more about that? Where does it hurt?"
- Assess severity conversationally: "Is it a little uncomfortable, or is it really bothering you?"
- Ask about impact: "Is the pain stopping you from doing anything you normally do?"

RESPONSE GUIDELINES:
- Acknowledge pain with empathy: "Oh, I'm sorry to hear that. That must be uncomfortable."
- For known chronic conditions (documented in patient record): Reference their condition naturally.
- For prescribed pain medication: Ask if they've taken it.
- NEVER suggest specific pain medications not in the patient record
- NEVER recommend dosage changes
- NEVER diagnose the cause of pain

ESCALATION:
- New or worsening pain: Flag as HIGH concern for caregiver
- Chest pain: Classify as CRITICAL — advise calling 911
- Pain preventing daily activities: Flag for physician review"""
    },
    {
        "id": "proto-005",
        "title": "Emotional Wellbeing and Loneliness Assessment Protocol",
        "category": "emotional",
        "source": "CareVoice Mental Health Guidelines",
        "keywords": ["emotional", "loneliness", "depression", "anxiety", "mental health", "isolation"],
        "content": """EMOTIONAL WELLBEING ASSESSMENT PROTOCOL

PURPOSE: Monitor emotional health and social isolation in elderly patients living independently.

ASSESSMENT APPROACH:
- Weave emotional check-ins naturally into conversation, not as clinical questions
- Ask: "How have your spirits been?" rather than "Are you depressed?"
- Listen for indicators: flat tone, short answers, expressions of hopelessness, withdrawal
- Ask about social contact using relationship and name from patient record

SIGNS OF CONCERN:
- Persistent sadness or flat affect across multiple calls
- Statements like "What's the point?" or "Nobody cares"
- Loss of interest in previously enjoyed activities
- Social withdrawal
- Changes in sleep or appetite concurrent with mood changes

RESPONSE GUIDELINES:
- Validate feelings: "I understand, that can feel really isolating. I'm glad we're chatting now."
- Do NOT minimize: Avoid "cheer up" or "it could be worse"
- Do NOT diagnose depression or anxiety
- Encourage connection: "It might feel good to give your daughter a call when you're up for it"

ESCALATION:
- Expressed hopelessness or suicidal ideation: CRITICAL — immediate alert
- Persistent low mood across 3+ consecutive calls: Flag for physician review
- Complete social isolation: Alert primary caregiver"""
    },
    {
        "id": "proto-006",
        "title": "Nutrition and Hydration Assessment Protocol",
        "category": "nutrition",
        "source": "CareVoice Clinical Guidelines — Nutrition",
        "keywords": ["nutrition", "meals", "hydration", "appetite", "eating", "diet"],
        "content": """NUTRITION AND HYDRATION ASSESSMENT PROTOCOL

PURPOSE: Monitor nutritional intake and hydration status in elderly patients.

ASSESSMENT APPROACH:
- Ask about recent meals: "Have you had anything nice to eat today?"
- Follow up on specifics naturally
- Ask about appetite: "Has your appetite been good lately?"
- Check hydration: "Have you been drinking enough water today?"

INDICATORS OF CONCERN:
- Skipping meals regularly
- Eating only one meal per day
- Significant appetite decrease
- Unintentional weight loss mentioned
- Difficulty preparing meals

RESPONSE GUIDELINES:
- Encourage without lecturing
- For patients with dietary restrictions: Reference their conditions from medical record naturally
- NEVER prescribe diets or supplements

ESCALATION:
- Multiple consecutive calls with skipped meals: Flag as MEDIUM concern
- Reported weight loss or inability to prepare food: Flag as HIGH concern
- Dehydration indicators: Flag as HIGH concern"""
    },
    {
        "id": "proto-007",
        "title": "Cognitive Screening During Wellness Calls",
        "category": "cognitive",
        "source": "CareVoice Clinical Guidelines — Cognitive Health",
        "keywords": ["cognitive", "memory", "confusion", "dementia", "screening", "orientation"],
        "content": """COGNITIVE SCREENING PROTOCOL

PURPOSE: Monitor cognitive function in elderly patients through natural conversation.

ASSESSMENT — OBSERVE NATURALLY:
- Orientation: Does the patient know what day it is, who called, what they had for breakfast?
- Memory: Do they remember their medications? Do they repeat questions?
- Language: Word-finding difficulties? Coherent sentences?
- Reasoning: Can they follow conversation flow?

RESPONSE GUIDELINES:
- If patient seems confused: Simplify language. Be patient.
- If patient repeats themselves: Respond naturally. Do NOT say "you already told me that."
- NEVER administer formal cognitive tests
- NEVER diagnose cognitive impairment

ESCALATION:
- Sudden confusion: Flag as HIGH — may indicate delirium or infection
- Progressive decline across multiple calls: Flag for physician review
- Unable to recognize caller or state own name: CRITICAL alert"""
    },
    {
        "id": "proto-008",
        "title": "Privacy and HIPAA Compliance Protocol",
        "category": "compliance",
        "source": "CareVoice Compliance Framework — HIPAA",
        "keywords": ["privacy", "HIPAA", "compliance", "PHI", "confidentiality", "identity"],
        "content": """PRIVACY AND HIPAA COMPLIANCE PROTOCOL

PURPOSE: Ensure all patient interactions comply with HIPAA privacy requirements.

IDENTITY VERIFICATION:
- Before ANY disclosure of patient information, verify identity
- Ask the person to state their first name — do not provide the patient's name first
- If identity cannot be verified after 2 attempts, end the call without disclosing PHI
- Log failed verification attempts as security events

PROTECTED HEALTH INFORMATION (PHI):
- PHI includes: name, conditions, medications, treatment plans, caregiver contacts, wellness scores
- PHI may ONLY be discussed after identity is verified
- AI responses referencing PHI must cite the source record

MINIMUM NECESSARY STANDARD:
- Only reference PHI relevant to the current conversation
- When reminding about medications, read exactly from the record

DOCUMENTATION AND AUDIT:
- Every response referencing PHI must be tracked with citation to source record
- Safety checks must verify no PHI leakage before identity verification
- All conversations logged for compliance review"""
    },
    {
        "id": "proto-009",
        "title": "Caregiver Alerting and Escalation Protocol",
        "category": "escalation",
        "source": "CareVoice Operations — Alert Management",
        "keywords": ["alert", "escalation", "caregiver", "SMS", "emergency", "notification"],
        "content": """CAREGIVER ALERTING AND ESCALATION PROTOCOL

PURPOSE: Define when and how to alert caregivers based on wellness check-in findings.

SEVERITY LEVELS:

CRITICAL (Immediate):
- Chest pain or breathing difficulty
- Fall with injury
- Suicidal ideation
- Severe confusion or disorientation
Action: Immediate SMS to primary caregiver. Flag for emergency services.

HIGH (Same-day):
- Pain preventing daily activities
- Fall without injury
- Missed medications for multiple days
- Significant appetite loss
- New confusion
Action: SMS to primary caregiver within 1 hour. Flag for physician.

MEDIUM (48 hours):
- Occasional missed medications
- Mild sleep disturbances
- Minor mood changes
Action: Include in daily caregiver summary.

LOW (Monitor):
- Stable condition
Action: Log for trend monitoring."""
    },
    {
        "id": "proto-010",
        "title": "AI Safety and Responsible Use Protocol",
        "category": "ai_safety",
        "source": "CareVoice Responsible AI Framework",
        "keywords": ["AI safety", "responsible AI", "guardrails", "hallucination", "grounding", "bias"],
        "content": """AI SAFETY AND RESPONSIBLE USE PROTOCOL

PURPOSE: Ensure the AI wellness companion operates safely and within defined guardrails.

GROUNDING REQUIREMENTS:
- All patient-specific claims must trace to the patient's medical record in Cosmos DB
- Care guidance must be grounded in approved protocols from Azure AI Search
- The AI must NEVER generate medical information not present in source documents
- Every response tracked with citation metadata linking claims to source documents

PRE-RESPONSE SAFETY GATE:
- Before any response reaches the patient, scan for dangerous content
- Blocked phrases: "you should take", "increase your dose", "your diagnosis is"
- If triggered, replace with safe fallback directing patient to care team
- All blocked responses logged for audit review

POST-RESPONSE SAFETY CHECKS:
- Medical guardrails: Detect unauthorized medical advice
- Elder respect: Flag ageist or condescending language
- Groundedness verification: Verify claims against patient record
- PHI protection: Ensure no data leaked before identity verification
- Azure Content Safety: Screen for hate, self-harm, sexual, or violent content

HALLUCINATION PREVENTION:
- System prompt restricts AI to information in patient record and approved protocols
- Citation tracking verifies which source fields each response references
- Post-call evaluation measures groundedness score (target: >80%)"""
    },
]


# ─── Upload with Embeddings ──────────────────────────────────────────────────

def upload_documents():
    docs_with_vectors = []

    for i, doc in enumerate(PROTOCOLS):
        print(f"  Embedding [{i+1}/{len(PROTOCOLS)}] {doc['title'][:50]}...")
        # Embed the content
        embed_text = f"{doc['title']}\n{doc['category']}\n{doc['content'][:2000]}"
        vector = get_embedding(embed_text)

        doc_with_vec = {
            "@search.action": "mergeOrUpload",
            **doc,
            "contentVector": vector,
        }
        docs_with_vectors.append(doc_with_vec)
        time.sleep(0.3)  # Rate limit courtesy

    # Upload batch
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version={API_VERSION_SEARCH}"
    r = requests.post(url, headers=HEADERS_SEARCH, json={"value": docs_with_vectors})

    if r.status_code in (200, 207):
        results = r.json().get("value", [])
        ok = sum(1 for v in results if v.get("status"))
        print(f"\n[OK] Uploaded {ok}/{len(PROTOCOLS)} documents with embeddings")
    else:
        print(f"\n[ERROR] Upload: {r.status_code} — {r.text[:200]}")
    return r.status_code in (200, 207)


# ─── Verify ──────────────────────────────────────────────────────────────────

def verify():
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version={API_VERSION_SEARCH}"
    r = requests.post(url, headers=HEADERS_SEARCH, json={
        "search": "medication",
        "select": "id,title,category,source",
        "top": 2,
    })
    if r.status_code == 200:
        results = r.json().get("value", [])
        print(f"\n[VERIFY] Text search for 'medication' → {len(results)} results:")
        for doc in results:
            print(f"  - {doc['title']} [{doc['category']}]")
    else:
        print(f"[VERIFY] Search failed: {r.status_code}")

    # Vector search test
    test_vector = get_embedding("patient forgot their medication what should I do")
    r2 = requests.post(url, headers=HEADERS_SEARCH, json={
        "vectorQueries": [{
            "kind": "vector",
            "vector": test_vector,
            "fields": "contentVector",
            "k": 2,
        }],
        "select": "id,title,category,source",
    })
    if r2.status_code == 200:
        results = r2.json().get("value", [])
        print(f"\n[VERIFY] Vector search for 'patient forgot medication' → {len(results)} results:")
        for doc in results:
            print(f"  - {doc['title']} [{doc['category']}] (score: {doc.get('@search.score', 'n/a')})")
    else:
        print(f"[VERIFY] Vector search failed: {r2.status_code} — {r2.text[:200]}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CareVoice AI — Vector RAG Index Setup")
    print("=" * 60)
    print(f"Search:    {SEARCH_ENDPOINT}")
    print(f"Index:     {INDEX_NAME}")
    print(f"Embedding: {OPENAI_ENDPOINT} / {EMBEDDING_DEPLOYMENT}")
    print()

    if create_index():
        print()
        print("Generating embeddings and uploading documents...")
        if upload_documents():
            verify()
            print("\n" + "=" * 60)
            print("DONE! Your vector RAG index is ready.")
            print("=" * 60)
    else:
        print("\nIndex creation failed. Check credentials.")