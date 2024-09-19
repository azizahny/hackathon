# pylint: disable=line-too-long,invalid-name
"""
This module demonstrates the usage of the Vertex AI Gemini 1.5 API within a Streamlit application.
"""

import os

import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_REGION")

vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models() -> tuple[GenerativeModel, GenerativeModel]:
    """Load Gemini 1.5 Flash and Pro models."""
    return GenerativeModel("gemini-1.5-flash"), GenerativeModel("gemini-1.5-pro")


def get_gemini_response(
    model: GenerativeModel,
    contents: str | list,
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=2048
    ),
    stream: bool = True,
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def get_model_name(model: GenerativeModel) -> str:
    """Get Gemini Model Name"""
    model_name = model._model_name.replace(  # pylint: disable=protected-access
        "publishers/google/models/", ""
    )
    return f"`{model_name}`"


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


st.header("Cakap Virtual Assistant", divider="rainbow")
gemini_15_flash, gemini_15_pro = load_models()

enterprise, upskill = st.tabs(
    ["Enterprise", "Upskill"]
)

with enterprise:
    st.subheader("Upskill syllabus generator")

    selected_model = st.radio(
        "Select Gemini Model:",
        [gemini_15_flash, gemini_15_pro],
        format_func=get_model_name,
        key="selected_model_story",
        horizontal=True,
    )

    # Upskill Syllabus Questionnaire

    company_name = st.text_input(
        "Enter company name: \n\n", key="company_name", value="Company Name"
    )
    company_industry = st.text_input(
        "Enter company industry: \n\n", key="company_industry", value="Company Industry"
    )
    job_title = st.text_input(
        "Enter job title: \n\n", key="job_title", value="Job Title"
    )
    job_level = st.text_input(
        "Enter job level: \n\n", key="job_level", value="Job Level"
    )
    class_type = st.radio(
        "Select the class type: \n\n",
        ["upskill", "language"],
        key="creative_control",
        horizontal=True,
    )
    if class_type == "language":
        language_class = st.multiselect(
            "What is the language that you need? (can select multiple) \n\n",
            [
                "English",
                "Mandarin",
                "Japanese",
                "Korean",
                "Bahasa Indonesia",
            ],
            key="language_skill"
        )
    elif class_type == "upskill":
        upskill_class = st.multiselect(
            "What is the upskill scope that you need? (can select multiple) \n\n",
            [
                "Business & Management",
                "Media & Creative",
                "Tourism & Hospitality",
                "Language",
                "Engineering",
                "Technology",
                "Career & Development",
                "Agriculture",
                "Green & Sustainability",
                "Education",
            ],
            key="upskill_class"
        )
    class_format = st.radio(
        "Select the class format: \n\n",
        ["offline", "online"],
        key="creative_control",
        horizontal=True,
    )
    learning_objective = st.text_input(
        "Enter your goal with this course: \n\n", key="learning_objective", value="Job Level"
    )

    prompt = f"""
    Write a comprehensive syllabus on the following premise:
    \n 
    company_industry: {company_industry}\n
    job_title: {job_title} \n
    job_level: {job_level} \n
    If the class_type: {class_type} then create upskill recommendation syllabus based on {upskill_class} type \n
    If the class_type: {class_type} then create language recommendation syllabus based on {language_class} type \n
    Provide a clear introduction to the course, outlining its objectives, learning outcomes, and the skills students will acquire.
    Divide the course into logical sections or modules. Each module should cover specific topics in detail and include subtopics as needed. Ensure the order of topics is coherent and follows a natural progression of learning.
    Specify the types of assessments (e.g., quizzes, assignments, projects) and how they align with learning outcomes. Include a grading rubric or percentage breakdown.      
    Provide a list of textbooks, articles, or other materials that students need to review. Include both mandatory and supplementary readings.
    Highlight any practical activities, labs, or case studies included in the syllabus to deepen understanding of the subject.
    """

    temperature = 0.95  # Default temperature value
    config = GenerationConfig(
        temperature=temperature  # No max_output_tokens specified
    )

    generate_t2t = st.button("Generate my syllabus", key="generate_t2t")
    if generate_t2t and prompt:
        with st.spinner(
            f"Generating your syllabus using {get_model_name(selected_model)} ..."
        ):
            first_tab1, first_tab2 = st.tabs(["Syllabus", "Prompt"])
            with first_tab1:
                response = get_gemini_response(
                    selected_model,  # Use the selected model
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Your syllabus:")
                    st.write(response)
            with first_tab2:
                st.text(
                    f"""Parameters:\n- Temperature: {temperature}\n"""
                )
                st.text(prompt)
