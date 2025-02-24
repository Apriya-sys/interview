import os
import streamlit as st
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
import speech_recognition as sr
from io import BytesIO
import tempfile

# Page configuration
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Load API Keys (Set up in .env file)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Define AI State
class InterviewState(TypedDict):
    job_description: str
    interview_questions: List[str]
    current_question: str
    answer: str
    feedback: str
    score: int
    final_feedback: str
    current_question_index: int
    max_questions: int
    interview_complete: bool
    previous_answers: List[dict]

# Step 1: Generate Questions
generate_questions_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""Based on the following job description, generate 5 interview questions in a numbered format:\n\n"
        "{job_description}\n\n"
        "Format the output as:\n"
        "1. [Question 1]\n"
        "2. [Question 2]\n"
        "3. [Question 3]\n"
        "4. [Question 4]\n"
        "5. [Question 5]"
 """
)
generate_questions_chain = generate_questions_prompt | llm

def generate_questions(state: InterviewState):
    response = generate_questions_chain.invoke({"job_description": state["job_description"]})
    questions = response.content.strip().split('\n')
    return {
        "interview_questions": questions, 
        "current_question": questions[0], 
        "current_question_index": 0,
        "max_questions": len(questions),
        "interview_complete": False,
        "previous_answers": []
    }

# Step 2: Analyze Answer
analyze_answer_prompt = PromptTemplate(
    input_variables=["current_question", "answer"],
    template="Evaluate this answer based on clarity, correctness, and depth.\n"
             "Question: {current_question}\n"
             "Answer: {answer}\n"
             "Provide a score out of 5 as a single number on a new line."
)
analyze_answer_chain = analyze_answer_prompt | llm

def analyze_answer(state: InterviewState):
    response = analyze_answer_chain.invoke({
        "current_question": state["current_question"], 
        "answer": state["answer"]
    })
    
    # Debugging LLM response
    st.write(f"Debug LLM Response: {response.content}")

    # Extract numeric score from response
    lines = response.content.strip().split('\n')
    score = None
    for line in reversed(lines):
        if line.strip().isdigit():  # Extract numeric value
            score = int(line.strip())
            break

    if score is None or score < 1 or score > 5:
        score = 3  # Default to neutral score if extraction fails

    state["score"] = score
    return state

# Step 3: Provide Feedback
feedback_prompt = PromptTemplate(
    input_variables=["answer", "score"],
    template="Provide constructive feedback on this answer based on its score ({score}/5).\nAnswer: {answer}"
)
feedback_chain = feedback_prompt | llm

def provide_feedback(state: InterviewState):
    response = feedback_chain.invoke({
        "answer": state["answer"], 
        "score": state["score"]
    })
    state["feedback"] = response.content
    state["previous_answers"].append({
        "question": state["current_question"],
        "answer": state["answer"],
        "feedback": state["feedback"],
        "score": state["score"]
    })
    return state

# Step 4: Next Question Logic
def next_question(state: InterviewState):
    new_index = state["current_question_index"] + 1
    if new_index < state["max_questions"]:
        state["current_question_index"] = new_index
        state["current_question"] = state["interview_questions"][new_index]
        state["answer"] = ""
    else:
        state["interview_complete"] = True
    return state

# Step 5: Final Feedback
final_feedback_prompt = PromptTemplate(
    input_variables=["previous_answers"],
    template="""Based on the interview performance, provide a final evaluation.
    
Previous answers and scores:
{previous_answers}

Give an overall assessment of the candidate's performance.
"""
)
final_feedback_chain = final_feedback_prompt | llm

def generate_final_feedback(state: InterviewState):
    previous_answers_text = "\n".join([
        f"Question {i+1}: {ans['question']}\n"
        f"Answer: {ans['answer']}\n"
        f"Score: {ans['score']}/5\n"
        for i, ans in enumerate(state["previous_answers"])
    ])
    
    response = final_feedback_chain.invoke({"previous_answers": previous_answers_text})
    state["final_feedback"] = response.content
    state["interview_complete"] = True
    return state

# Speech Recognition
def recognize_speech_from_mic(audio_file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio.flush()
        temp_audio.seek(0)
        with sr.AudioFile(temp_audio) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.RequestError:
        return "API unavailable"
    except sr.UnknownValueError:
        return "Unable to recognize speech"

# Helper function to style score
def get_score_class(score):
    if score >= 4:
        return "✅ High Score"
    elif score >= 3:
        return "⚠️ Medium Score"
    else:
        return "❌ Low Score"

# Initialize session state
if "interview_state" not in st.session_state:
    st.session_state.interview_state = {
        "job_description": "",
        "interview_questions": [],
        "current_question": "",
        "answer": "",
        "feedback": "",
        "score": 0,
        "final_feedback": "",
        "current_question_index": 0,
        "max_questions": 0,
        "interview_complete": False,
        "previous_answers": []
    }

# UI Components
st.title("🎙️ AI Interview Assistant")

if not st.session_state.interview_state["interview_questions"]:
    job_description = st.text_area("Enter Job Description:", "Looking for a Python Developer with Flask, SQL, and REST API skills.")

    if st.button("Start Interview"):
        state = generate_questions({"job_description": job_description})
        st.session_state.interview_state.update(state)
        st.rerun()

else:
    state = st.session_state.interview_state
    st.subheader(f"Question {state['current_question_index'] + 1}: {state['current_question']}")

    if not state["answer"]:
        answer_text = st.text_area("Your Answer:")
        if st.button("Submit Answer"):
            state["answer"] = answer_text
            state = analyze_answer(state)
            state = provide_feedback(state)
            st.session_state.interview_state.update(state)
            st.rerun()
    else:
        st.write(f"**Your Answer:** {state['answer']}")
        st.write(f"**Feedback:** {state['feedback']}")
        st.write(f"**Score:** {state['score']}/5 - {get_score_class(state['score'])}")

        if st.button("Next Question"):
            state = next_question(state)
            st.session_state.interview_state.update(state)
            st.rerun()

# Final Feedback
if state["interview_complete"]:
    st.subheader("🏆 Final Evaluation")
    st.write(state["final_feedback"])
    if st.button("Restart Interview"):
        st.session_state.interview_state = {}
        st.rerun()
