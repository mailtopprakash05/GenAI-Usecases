Healthcare Assistant — User Guide
=================================

This document explains how the Healthcare Assistant app works, how to ask
questions to get the best answers, and provides sample question + answer
pairs you can use as references or paste directly into the app.

1. Overview
-----------
The Healthcare Assistant is a Retrieval-Augmented Generation (RAG) demo:
- It reads PDF documents from `Healthcare_Assistant/data/`.
- Chunks the text into manageable pieces.
- Indexes them using FAISS + OpenAI embeddings when available.
- If embeddings or FAISS cannot be used, a local TF-IDF fallback enables
  simple retrieval so the app remains responsive.
- Retrieved chunks are sent to OpenAI's Chat Completion API with the user's
  prompt to produce a synthesized answer.

2. How to use the app (step-by-step)
-----------------------------------
1. Start the app (see README for setup). Open the Streamlit URL (usually
   `http://localhost:8501`).
2. In the chat input, type your question. Keep it specific and supply a
   bit of context when relevant (age, comorbidity, outpatient vs inpatient).
3. Send the query. The app will search the indexed documents and produce an
   answer that is based on the texts found.
4. If you need source references, append: "Please cite the document names
   or include short quotes from the source." The app will include sources
   if available.

3. Best practices for asking questions
-------------------------------------
- Be explicit: name conditions, timeframes, or populations (e.g., "adult",
  "pregnant", "pediatric").
- Prefer one main question per prompt; use follow-ups for clarification.
- If you want patient-facing language, ask: "Explain like I'm a patient".
- To receive clinician-level detail, ask: "Explain for a clinician".
- Ask to list references if you need to know which PDF the answer came from.

4. Example prompt templates
---------------------------
- "Summarize the recommended screening intervals for colorectal cancer."
- "What initial tests should be ordered for suspected urinary tract infection
  in an adult?" 
- "Provide a 5-step plan for lifestyle changes to manage hypertension."
- "Compare first-line treatments for mild vs severe psoriasis and cite
  sources." 

5. Sample Q&A (copy-and-use)
----------------------------
Below are 12 sample questions with model-style answers. These are examples
showing how the assistant synthesizes retrieved text into helpful replies.

Q1: "What are the most common symptoms of type 2 diabetes?"
A1: "Common symptoms include increased thirst, frequent urination,
blurred vision, fatigue and unexplained weight loss. For many patients the
onset is gradual and some people may be asymptomatic; screening for
high-risk patients is recommended. (See source: 'common_diseases.pdf')"

Q2: "List 5 lifestyle changes recommended to help manage hypertension."
A2: "1) Reduce dietary sodium; 2) Adopt the DASH diet with more fruits and
vegetables; 3) Increase regular aerobic physical activity (e.g., 150 min
per week); 4) Limit alcohol consumption; 5) Maintain a healthy weight and
manage stress. Monitor blood pressure regularly. (Source: 'preventive_care.pdf')"

Q3: "What are red flags for back pain that require urgent evaluation?"
A3: "Red flags include: fever or signs of systemic infection, progressive
neurologic deficits (e.g., foot drop), loss of bowel/bladder control,
history of malignancy, unexplained weight loss, or recent significant
trauma. Patients with these signs should be evaluated promptly in urgent
care or emergency settings. (Source: 'treatments.pdf')"

Q4: "How is community-acquired pneumonia (CAP) initially managed in adults?"
A4: "Initial management usually includes assessment of severity (use of
CURB-65 or clinical judgment), outpatient vs inpatient decision, early
empiric antibiotics covering typical and atypical organisms, oxygen if
hypoxic, and supportive care (fluids, antipyretics). Antibiotic choice is
based on local guidance and patient comorbidities. (Source: 'treatments.pdf')"

Q5: "What vaccinations are recommended for adults over 65?"
A5: "Common recommendations include influenza vaccine annually, pneumococcal
vaccination according to current immunization schedules (PCV and/or PPSV),
and shingles (herpes zoster) vaccination if not previously given. Check
local immunization guidelines for the exact schedule. (Source: 'wellness.pdf')"

Q6: "Give a short, patient-friendly explanation of what hypertension is."
A6: "Hypertension, or high blood pressure, means the pressure of blood in
your arteries is persistently higher than normal. Over time it can raise
your risk for heart attack, stroke and kidney disease. Lifestyle changes
and medications can control it — work with your provider to find the best
plan for you. (Patient-friendly paraphrase from 'preventive_care.pdf')"

Q7: "What first-line treatment is suggested for mild eczema?"
A7: "For mild eczema, basic skin care (emollients/moisturizers) and topical
low-to-medium potency corticosteroids for flare control are commonly
recommended. Identify and avoid triggers and consider referral if
worsening. (Source: 'treatments.pdf')"

Q8: "Describe a 7-day diet plan for general wellness after minor surgery."
A8: "A 7-day supportive plan emphasizes protein for healing, hydration,
small frequent meals if appetite is low, fiber to prevent constipation,
and vitamin-rich fruits/vegetables. Avoid alcohol and heavy meals in the
first 48 hours if nausea occurs. (This is a synthesis of 'wellness.pdf')"

Q9: "What screening is recommended for type 2 diabetes in adults?"
A9: "Screen adults with obesity or other risk factors (family history,
hypertension, dyslipidemia) starting at age suggested by local guidance;
commonly testing uses fasting glucose, HbA1c, or 2-hour glucose tolerance
as indicated. (Source: 'preventive_care.pdf')"

Q10: "How should suspected opioid overdose be managed acutely?"
A10: "Provide immediate basic life support as needed, call emergency
services, administer naloxone if opioid overdose is suspected and
available, and monitor airway and breathing. After stabilization, arrange
transport to an ED for further care. (Source: emergency protocol in 'treatments.pdf')"

Q11: "What are common side effects of metformin and monitoring
recommendations?"
A11: "Common side effects include gastrointestinal upset (nausea,
diarrhea). Monitor renal function prior to initiation and periodically,
and assess vitamin B12 levels during long-term use. Adjust dosing for
renal impairment according to guidelines. (Source: 'treatments.pdf')"

Q12: "Provide quick counseling points for patients starting statin
therapy."
A12: "1) Explain the purpose: reduce cardiovascular risk; 2) Take as
prescribed (usually at night for some statins); 3) Report persistent
muscle pain or unexplained fatigue; 4) Maintain healthy diet and activity;
5) Periodic liver function tests and lipid panel monitoring as advised.
(Source: 'preventive_care.pdf')"

6. When results are missing or low-confidence
-------------------------------------------
- If the assistant cannot find relevant text, it may synthesize a general
  answer from its instruction context. In that case, follow up with a
  narrower question or check that relevant PDFs exist in `data/`.
- If you see the TF-IDF fallback is being used (or if embedding calls fail),
  add more domain-specific PDFs to improve retrieval quality.

7. Want these sample Q&A in the app UI?
--------------------------------------
I can add a small sidebar widget with clickable example prompts that paste
into the input. Tell me if you'd like that and whether you want all 12
examples or a smaller curated set.
