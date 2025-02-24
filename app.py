import os
import streamlit as st
import speech_recognition as sr
import tempfile
import time
from typing_extensions import TypedDict
from typing import List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate

# Configure Streamlit page
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API Keys
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Define Interview State
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
    interview_mode: str

# Sidebar: Select Interview Mode
st.sidebar.markdown("## Interview Mode")
interview_mode = st.sidebar.selectbox(
    "Choose an interview mode:",
    ["Practice Mode", "Mock Mode", "Challenge Mode"],
    index=0
)

# Step 1: Generate Questions
generate_questions_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""Based on the following job description, generate 5 diverse interview questions:
    1. 2 Technical questions (specific to skills)
    2. 2 Behavioral questions (STAR format)
    3. 1 Situational question (hypothetical problem-solving)
    
    Job Description:
    {job_description}
    
    Format:
    1. [Question]
    2. [Question]
    3. [Question]
    4. [Question]
    5. [Question]
    """
)
generate_questions_chain = generate_questions_prompt | llm

def generate_questions(state: InterviewState):
    response = generate_questions_chain.invoke({"job_description": state["job_description"]})
    questions = response.content.strip().split("\n")
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
    template="Evaluate this answer (1-5 scale) based on clarity, correctness, and depth.\nQuestion: {current_question}\nAnswer: {answer}\nReturn score only."
)
analyze_answer_chain = analyze_answer_prompt | llm

def analyze_answer(state: InterviewState):
    response = analyze_answer_chain.invoke({
        "current_question": state["current_question"], 
        "answer": state["answer"]
    })
    score = int(response.content.strip())
    state["score"] = score
    return state

# Step 3: Provide Feedback
feedback_prompt = PromptTemplate(
    input_variables=["answer", "score"],
    template="Provide constructive feedback based on this answer's score ({score}/5).\nAnswer: {answer}"
)
feedback_chain = feedback_prompt | llm

def provide_feedback(state: InterviewState):
    response = feedback_chain.invoke({
        "answer": state["answer"], 
        "score": state["score"]
    })
    new_previous_answers = state.get("previous_answers", []).copy()
    new_previous_answers.append({
        "question": state["current_question"],
        "answer": state["answer"],
        "feedback": response.content,
        "score": state["score"]
    })
    state["feedback"] = response.content
    state["previous_answers"] = new_previous_answers
    return state

# Step 4: Route Next Step
def route_after_feedback(state: InterviewState):
    if state["current_question_index"] >= state["max_questions"] - 1:
        return "finish"
    return "next"

def next_question(state: InterviewState):
    new_index = state["current_question_index"] + 1
    state["current_question_index"] = new_index
    if new_index < len(state["interview_questions"]):
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
        f"Q{i+1}: {ans['question']}\nA: {ans['answer']}\nScore: {ans['score']}/5"
        for i, ans in enumerate(state["previous_answers"])
    ])
    response = final_feedback_chain.invoke({"previous_answers": previous_answers_text})
    return {
        "final_feedback": response.content,
        "interview_complete": True
    }

# Speech Recognition
def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio.flush()
        temp_audio.seek(0)
        with sr.AudioFile(temp_audio) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "Speech recognition service unavailable."

# Build Workflow
workflow = StateGraph(InterviewState)
workflow.add_node("generate_questions", generate_questions)
workflow.add_edge(START, "generate_questions")
workflow.add_edge("generate_questions", END)

answer_workflow = StateGraph(InterviewState)
answer_workflow.add_node("analyze_answer", analyze_answer)
answer_workflow.add_node("provide_feedback", provide_feedback)
answer_workflow.add_node("next_question", next_question)
answer_workflow.add_node("generate_final_feedback", generate_final_feedback)
answer_workflow.add_edge(START, "analyze_answer")
answer_workflow.add_edge("analyze_answer", "provide_feedback")
answer_workflow.add_conditional_edges(
    "provide_feedback",
    route_after_feedback,
    {"next": "next_question", "finish": "generate_final_feedback"}
)
answer_workflow.add_edge("next_question", END)
answer_workflow.add_edge("generate_final_feedback", END)

interview_graph = workflow.compile()
answer_graph = answer_workflow.compile()

# Session State Management
if "interview_state" not in st.session_state:
    st.session_state.interview_state = None
    st.session_state.submitted_answer = False

st.markdown("## 🎙️ AI Interview Assistant")
job_description = st.text_area("Enter job description:", "Looking for a Python Developer with experience in Flask and SQL.")

if st.button("🚀 Start Interview"):
    initial_state = {
        "job_description": job_description,
        "interview_questions": [],
        "current_question": "",
        "answer": "",
        "feedback": "",
        "score": 0,
        "final_feedback": "",
        "current_question_index": 0,
        "max_questions": 0,
        "interview_complete": False,
        "previous_answers": [],
        "interview_mode": interview_mode
    }
    interview_state = interview_graph.invoke(initial_state)
    st.session_state.interview_state = interview_state
    st.rerun()

# Display questions, answers, and feedback as in the previous version...
# Main content area
if st.session_state.interview_state:
    state = st.session_state.interview_state

    if state.get("interview_complete", False):
        # Final evaluation section
        st.markdown("## 🏆 Interview Complete!")
        
        # Calculate average score
        total_score = sum(ans["score"] for ans in state["previous_answers"])
        avg_score = total_score / len(state["previous_answers"]) if state["previous_answers"] else 0
        
        # Display overall score
        st.markdown(f"### Overall Score: **{avg_score:.1f}/5**")
        
        # Final feedback
        st.markdown("### Final Evaluation")
        st.write(state["final_feedback"])

        # Interview summary
        st.markdown("## Interview Summary")
        for i, ans in enumerate(state["previous_answers"]):
            with st.expander(f"Question {i+1}: {ans['question']}"):
                st.write(f"**Your Answer:** {ans['answer']}")
                st.write(f"**Score:** {ans['score']}/5")
                st.write(f"**Feedback:** {ans['feedback']}")

        # Restart Interview Button
        if st.button("🔄 Start New Interview"):
            st.session_state.interview_state = None
            st.session_state.submitted_answer = False
            st.rerun()

    else:
        # Interview in progress
        curr_idx = state["current_question_index"]
        max_questions = state["max_questions"]

        # Progress indicator
        st.progress((curr_idx + 1) / max_questions)

        # Display current question
        st.markdown(f"### Question {curr_idx + 1} of {max_questions}")
        st.markdown(f"**{state['current_question']}**")

        # Answer submission section
        if not st.session_state.submitted_answer:
            st.markdown("### Speak Your Answer:")
            audio_data = st.audio_input("Record your answer")

            if audio_data:
                st.audio(audio_data, format="audio/wav")
                with st.spinner("Transcribing your answer..."):
                    transcribed_text = recognize_speech(audio_data)

                st.markdown("### Transcribed Answer:")
                edited_text = st.text_area("Edit your response before submitting", transcribed_text)

                # Submit Answer
                if st.button("Submit Answer"):
                    st.session_state.answer_text = edited_text
                    eval_state = state.copy()
                    eval_state["answer"] = edited_text

                    with st.spinner("Analyzing your answer..."):
                        updated_state = answer_graph.invoke(eval_state)

                    st.session_state.interview_state = updated_state
                    st.session_state.submitted_answer = True
                    st.rerun()
        
        else:
            # Display previous answer, feedback, and score
            st.markdown("### Your Answer:")
            st.write(st.session_state.answer_text)

            st.markdown("### Feedback:")
            st.write(state["feedback"])

            st.markdown(f"### Score: **{state['score']}/5**")

            # Continue to Next Question Button
            if st.button("Continue to Next Question" if curr_idx < max_questions - 1 else "Complete Interview"):
                if curr_idx < max_questions - 1:
                    with st.spinner("Preparing next question..."):
                        st.session_state.interview_state = state
                        st.session_state.submitted_answer = False
                else:
                    with st.spinner("Generating final evaluation..."):
                        final_state = generate_final_feedback(state.copy())
                        updated_state = state.copy()
                        updated_state.update(final_state)
                        st.session_state.interview_state = updated_state
                st.rerun()

# Footer
st.markdown("---")
st.markdown("🎙️ AI Interview Assistant © 2025 | Powered by LangChain & Groq")
